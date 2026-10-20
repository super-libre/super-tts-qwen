// SPDX-License-Identifier: GPL-3.0-only
//! The one thread the model lives on.
//!
//! [`QwenTts`](crate::model::QwenTts) is not `Send`: Burn's generation keeps
//! its frame and captured-graph state in `Rc<RefCell<..>>`, deliberately, since
//! that state is threaded through a single sequential decode loop and an atomic
//! refcount would be paid on every one of the thirteen thousand operations a
//! frame goes through. So the model cannot sit behind a mutex in the shared
//! state the handlers hold, and it cannot be moved onto a `spawn_blocking`
//! thread either.
//!
//! It gets a thread of its own instead, and every use of it is a job sent
//! there. That also gives the serialization a mutex would: jobs run one at a
//! time, in the order they were submitted, which is what a single GPU and a
//! model with one key/value cache want anyway.
//!
//! And it knows when the model is idle: once no job has come for
//! [`IDLE_RELEASE`], it hands the device memory the last requests grew back to
//! the driver.

use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::Duration;

use crate::model::QwenTts;

/// How long the model thread waits for the next job before it releases the
/// memory the last ones left behind, see [`QwenTts::release_memory`].
///
/// The daemon sends an utterance one sentence at a time, each request as soon
/// as the last one ends, so a gap this long is the end of what it was reading.
/// The release takes tens of milliseconds, and the next request about as much
/// to allocate again.
const IDLE_RELEASE: Duration = Duration::from_secs(30);

/// Work to run on the model thread. `None` before the first successful load,
/// which is why a job takes the slot rather than the model.
type Job = Box<dyn FnOnce(&mut Option<QwenTts>) + Send>;

/// A handle to the model thread. Cloneable, `Send + Sync`, and the only way to
/// reach the model.
#[derive(Debug)]
pub struct ModelThread {
    jobs: Sender<Job>,
}

/// What the model thread does next.
enum Next {
    Run(Job),
    /// Nothing came for [`IDLE_RELEASE`].
    Release,
    /// Every sender is gone.
    Stop,
}

/// Waits for the next job, for at most `idle` when `used`: whether a job ran
/// since the memory was last released.
fn next(jobs: &Receiver<Job>, used: bool, idle: Duration) -> Next {
    if !used {
        return jobs.recv().map_or(Next::Stop, Next::Run);
    }
    match jobs.recv_timeout(idle) {
        Ok(job) => Next::Run(job),
        Err(RecvTimeoutError::Timeout) => Next::Release,
        Err(RecvTimeoutError::Disconnected) => Next::Stop,
    }
}

impl ModelThread {
    /// Start the thread. It owns the model slot and lives until the sender is
    /// dropped, which happens when the process is shutting down.
    #[must_use]
    pub fn spawn() -> Self {
        let (jobs, rx) = channel::<Job>();
        std::thread::Builder::new()
            .name("qwen-model".to_string())
            .spawn(move || {
                let mut model: Option<QwenTts> = None;
                let mut used = false;
                // Blocking is correct here and only here: this is a plain
                // thread, not one of the runtime's, so blocking it parks
                // nothing the runtime needed.
                loop {
                    let job = match next(&rx, used, IDLE_RELEASE) {
                        Next::Run(job) => job,
                        Next::Release => {
                            if let Some(model) = &model {
                                log::debug!("idle; releasing the device memory");
                                model.release_memory();
                            }
                            used = false;
                            continue;
                        }
                        Next::Stop => break,
                    };
                    used = true;
                    // A panic here would otherwise end the thread and leave a
                    // process that still answers `/v1/ping` while every real
                    // request fails — the one failure the daemon cannot see,
                    // because a backend that exits is restarted and one that
                    // lies is not.
                    //
                    // The model is dropped rather than kept: a generation that
                    // unwound half way through leaves the key/value cache and
                    // the captured graphs describing a step that never
                    // finished, and answering the next request from that is
                    // worse than refusing it. `POST /v1/load` builds a fresh
                    // one.
                    let run = std::panic::AssertUnwindSafe(|| job(&mut model));
                    if std::panic::catch_unwind(run).is_err() {
                        log::error!("a model job panicked; dropping the model");
                        model = None;
                    }
                }
                log::debug!("the model thread is done");
            })
            // A backend that cannot start its model thread can serve `/v1/ping`
            // and nothing else, which is a worse failure than not starting.
            .expect("spawning the model thread");
        Self { jobs }
    }

    /// Queue `job` and return without waiting for it.
    ///
    /// Returns `false` if the model thread is gone. Used for the two jobs whose
    /// results reach the caller some other way: a load, which reports through
    /// `GET /v1/status`, and a synthesis, which streams into the response body
    /// that has already been sent.
    pub fn submit(&self, job: impl FnOnce(&mut Option<QwenTts>) + Send + 'static) -> bool {
        // Never blocks: the channel is unbounded.
        self.jobs.send(Box::new(job)).is_ok()
    }

    /// Queue `job` and wait for what it returns.
    ///
    /// `None` if the model thread is gone or the job was dropped without
    /// running — neither is recoverable, and both are reported as a backend
    /// that is not ready.
    pub async fn run<T: Send + 'static>(
        &self,
        job: impl FnOnce(&mut Option<QwenTts>) -> T + Send + 'static,
    ) -> Option<T> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.jobs
            .send(Box::new(move |model| {
                // The receiver is gone when the request was abandoned while
                // queued. The job still ran; there is just nobody to tell.
                let _ = tx.send(job(model));
            }))
            .ok()?;
        rx.await.ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A panicking job must not take the thread with it. Losing the thread
    /// would leave a process that still answers `/v1/ping` while every real
    /// request fails, which is the one failure the daemon cannot act on: a
    /// backend that exits is restarted, one that lies is not.
    ///
    /// The panic prints its message and a backtrace note to stderr while this
    /// runs. That is the panic hook doing its job, not a failure.
    #[tokio::test]
    async fn a_panicking_job_does_not_take_the_thread_with_it() {
        let thread = ModelThread::spawn();
        assert!(thread.submit(|_| panic!("a job went wrong")));
        // Answering at all is the assertion: the thread had to survive the
        // panic to run this.
        assert_eq!(thread.run(|slot| slot.is_none()).await, Some(true));
    }

    /// Jobs run in the order they were submitted — the serialization the model
    /// needs, since one key/value cache cannot serve two generations at once.
    #[tokio::test]
    async fn jobs_run_in_order() {
        use std::sync::{Arc, Mutex};

        let thread = ModelThread::spawn();
        let seen = Arc::new(Mutex::new(Vec::new()));
        for i in 0..8 {
            let seen = Arc::clone(&seen);
            assert!(thread.submit(move |_| seen.lock().unwrap().push(i)));
        }
        thread.run(|_| ()).await.expect("the thread answers");
        assert_eq!(*seen.lock().unwrap(), (0..8).collect::<Vec<i32>>());
    }

    /// The thread releases memory once it has gone `idle` without a job after
    /// one ran, and not before.
    #[test]
    fn an_idle_thread_releases_once_a_job_has_run() {
        let idle = Duration::from_millis(20);
        let (jobs, rx) = channel::<Job>();
        jobs.send(Box::new(|_| ())).unwrap();
        assert!(matches!(next(&rx, false, idle), Next::Run(_)));
        jobs.send(Box::new(|_| ())).unwrap();
        assert!(matches!(next(&rx, true, idle), Next::Run(_)));
        assert!(matches!(next(&rx, true, idle), Next::Release));
        drop(jobs);
        assert!(matches!(next(&rx, true, idle), Next::Stop));
        assert!(matches!(next(&rx, false, idle), Next::Stop));
    }
}

//! Scrubbing by grain restart, as in losslessaudiocut: while the user drags
//! with the right mouse button, playback restarts at the pointer at most
//! every [`GRAIN_INTERVAL`]. Between restarts it plays normally, so one
//! hears short undistorted pieces of the (processed) audio and can find a
//! call by ear; when the pointer stops, playback simply continues there.
//!
//! The player fades at every jump (see `player.rs`), so the grains join
//! without clicks. This module only decides *when* to restart *where*; it
//! takes the time as a parameter, so it is tested without waiting.

use std::time::{Duration, Instant};

/// Shortest time between two restarts. Shorter grains sound like a buzz;
/// longer ones make the sound lag behind the pointer.
pub const GRAIN_INTERVAL: Duration = Duration::from_millis(40);

/// Decides when a scrub restarts playback.
#[derive(Debug, Clone)]
pub struct Scrubber {
    /// Shortest time between two restarts.
    interval: Duration,
    /// When playback was last restarted, and where (sample).
    last: Option<(Instant, usize)>,
    /// A position the pointer moved to too soon after the last restart.
    pending: Option<usize>,
}

impl Default for Scrubber {
    /// A scrubber restarting at most every [`GRAIN_INTERVAL`].
    fn default() -> Self {
        Scrubber::new(GRAIN_INTERVAL)
    }
}

impl Scrubber {
    /// A scrubber restarting at most every `interval`.
    pub fn new(interval: Duration) -> Self {
        Scrubber {
            interval,
            last: None,
            pending: None,
        }
    }

    /// The pointer moved to sample `pos` at `now`. Returns where to restart
    /// playback now, or `None` (too soon, or no movement); a position that
    /// came too soon is restarted at by [`Scrubber::tick`].
    pub fn move_to(&mut self, pos: usize, now: Instant) -> Option<usize> {
        match self.last {
            Some((_, at)) if at == pos => {
                // Back where playback last restarted: nothing to do.
                self.pending = None;
                None
            }
            Some((t, _)) if now.saturating_duration_since(t) < self.interval => {
                self.pending = Some(pos);
                None
            }
            _ => self.restart(pos, now),
        }
    }

    /// Timer: restarts at the latest position that came too soon, once the
    /// interval has passed, so playback ends up where the pointer stopped.
    pub fn tick(&mut self, now: Instant) -> Option<usize> {
        let pos = self.pending?;
        let due = self
            .last
            .is_none_or(|(t, _)| now.saturating_duration_since(t) >= self.interval);
        if due {
            self.restart(pos, now)
        } else {
            None
        }
    }

    /// Where the pointer is, as far as the scrub knows (the pending
    /// position, else the last restart).
    pub fn position(&self) -> Option<usize> {
        self.pending.or(self.last.map(|(_, p)| p))
    }

    /// Records a restart at `pos` and returns it.
    fn restart(&mut self, pos: usize, now: Instant) -> Option<usize> {
        self.last = Some((now, pos));
        self.pending = None;
        Some(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` milliseconds.
    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn restarts_at_most_every_interval_and_catches_up() {
        let t0 = Instant::now();
        let mut s = Scrubber::new(ms(40));
        assert_eq!(s.position(), None);
        assert_eq!(s.move_to(1000, t0), Some(1000), "the first move restarts");
        // Moves within the interval wait.
        assert_eq!(s.move_to(1100, t0 + ms(10)), None);
        assert_eq!(s.move_to(1200, t0 + ms(30)), None);
        assert_eq!(s.position(), Some(1200));
        assert_eq!(s.tick(t0 + ms(39)), None);
        // The timer restarts at the latest position once it is due.
        assert_eq!(s.tick(t0 + ms(40)), Some(1200));
        assert_eq!(s.tick(t0 + ms(100)), None, "nothing pending");
        // A move after the interval restarts at once.
        assert_eq!(s.move_to(900, t0 + ms(80)), Some(900));
    }

    #[test]
    fn no_movement_no_restart() {
        let t0 = Instant::now();
        let mut s = Scrubber::default();
        assert_eq!(s.move_to(500, t0), Some(500));
        // The pointer stands still: playback goes on normally.
        assert_eq!(s.move_to(500, t0 + ms(200)), None);
        // Moving away and back within the interval cancels the restart.
        assert_eq!(s.move_to(600, t0 + ms(210)), Some(600));
        assert_eq!(s.move_to(700, t0 + ms(220)), None);
        assert_eq!(s.move_to(600, t0 + ms(230)), None);
        assert_eq!(s.tick(t0 + ms(300)), None);
        assert_eq!(s.position(), Some(600));
    }
}

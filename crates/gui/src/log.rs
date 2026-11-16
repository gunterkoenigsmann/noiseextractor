//! The log pane: every message the status bar showed, with the time of
//! day, kept to a bounded number of lines (as in losslessaudiocut).
//!
//! The status bar shows one message at a time and the next one replaces
//! it; the log keeps them, so a result that flashed by while the user
//! looked elsewhere (which partials were found, how often the sound
//! repeats) can still be read.
//!
//! Messages come from many places, some of them while the application
//! state is borrowed, so the pane's control lives in a thread-local of the
//! GUI thread rather than in that state.

use std::cell::RefCell;
use std::collections::VecDeque;
use wxdragon::prelude::*;
use wxdragon::DateTime;

/// What the log's text control has to do after [`Log::push`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Update {
    /// Nothing (an empty message, or the same as the last one).
    None,
    /// Append this line (it ends in a newline).
    Append(String),
    /// The log was full and dropped its oldest lines: replace the whole
    /// text with this.
    Replace(String),
}

/// The messages, oldest first.
#[derive(Debug, Clone)]
pub struct Log {
    /// One line per message, each ending in a newline.
    lines: VecDeque<String>,
    /// The text of the last message, without its time, to drop repeats.
    last: Option<String>,
    /// At most this many lines are kept.
    max: usize,
}

impl Log {
    /// An empty log keeping at most `max` lines (at least 2).
    pub fn new(max: usize) -> Self {
        Log {
            lines: VecDeque::new(),
            last: None,
            max: max.max(2),
        }
    }

    /// Adds `text`, shown at `hour:minute:second`. A message equal to the
    /// previous one is left out (a timer may report the same state again).
    /// When the log is full, the oldest half goes, so the text control is
    /// rewritten only now and then.
    pub fn push(&mut self, (h, m, s): (i32, i32, i32), text: &str) -> Update {
        let text = text.trim();
        if text.is_empty() || self.last.as_deref() == Some(text) {
            return Update::None;
        }
        self.last = Some(text.to_string());
        let line = format!("{h:02}:{m:02}:{s:02}  {text}\n");
        self.lines.push_back(line.clone());
        if self.lines.len() <= self.max {
            return Update::Append(line);
        }
        let keep = self.max / 2;
        let drop = self.lines.len() - keep;
        self.lines.drain(..drop);
        Update::Replace(self.lines.iter().map(String::as_str).collect())
    }
}

/// Lines kept; the oldest half goes when there are more.
const MAX_LINES: usize = 2000;

thread_local! {
    /// The pane's control and its lines, once created.
    static PANE: RefCell<Option<(TextCtrl, Log)>> = const { RefCell::new(None) };
}

/// Creates the pane's text control as a child of `parent`.
pub fn create(parent: &Frame) -> TextCtrl {
    let ctrl = TextCtrl::builder(parent)
        .with_style(TextCtrlStyle::MultiLine | TextCtrlStyle::ReadOnly | TextCtrlStyle::DontWrap)
        .build();
    PANE.with(|p| *p.borrow_mut() = Some((ctrl, Log::new(MAX_LINES))));
    ctrl
}

/// Adds `text` to the log, if the pane exists.
pub fn append(text: &str) {
    let now = DateTime::now();
    let time = (now.hour() as i32, now.minute() as i32, now.second() as i32);
    PANE.with(|p| {
        let Some((ctrl, log)) = &mut *p.borrow_mut() else {
            return;
        };
        match log.push(time, text) {
            Update::None => {}
            // Appending scrolls to the end, so the newest line shows.
            Update::Append(line) => ctrl.append_text(&line),
            Update::Replace(all) => {
                ctrl.change_value("");
                ctrl.append_text(&all);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_carry_the_time_and_repeats_are_dropped() {
        let mut log = Log::new(10);
        assert_eq!(
            log.push((9, 5, 7), "  Loaded birds.wav "),
            Update::Append("09:05:07  Loaded birds.wav\n".into())
        );
        assert_eq!(log.push((9, 5, 8), "Loaded birds.wav"), Update::None);
        assert_eq!(log.push((9, 5, 8), "   "), Update::None);
        assert!(matches!(log.push((9, 5, 9), "Saved"), Update::Append(_)));
    }

    #[test]
    fn a_full_log_drops_its_older_half() {
        let mut log = Log::new(4);
        for i in 0..4 {
            assert!(matches!(log.push((0, 0, i), &format!("m{i}")), Update::Append(_)));
        }
        assert_eq!(
            log.push((0, 0, 4), "m4"),
            Update::Replace("00:00:03  m3\n00:00:04  m4\n".into())
        );
    }
}

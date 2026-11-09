#[cfg(any(debug_assertions, test))]
use std::cmp;
use std::fmt;
use std::fmt::Display;
#[cfg(any(debug_assertions, test))]
use std::time::{Duration, Instant};

use crate::solver::grid::DirectionSet;

#[cfg(any(debug_assertions, test))]
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Metrics {
    games_played: usize,
    solve_passes: usize,
    max_partitions: usize,
    max_partition_size: usize,
    partitions_created: usize,
    neighbor_visits: usize,
    indexing_duration: Duration,
    desolved_duration: Duration,
    basic_solve_duration: Duration,
    partition_duration: Duration,
    mark_time: Option<Instant>,
}

#[cfg(any(debug_assertions, test))]
impl Metrics {
    pub(super) fn games_played(&self) -> usize {
        self.games_played
    }

    #[cfg(test)]
    pub(super) fn tally_game(&mut self, other: &Metrics) {
        self.games_played += 1;
        self.solve_passes += other.solve_passes();
        self.max_partitions = cmp::max(self.max_partitions, other.max_partitions());
        self.max_partition_size = cmp::max(self.max_partition_size, other.max_partition_size());
        self.partitions_created += other.partitions_created();
        self.neighbor_visits += other.neighbor_visits();
        self.indexing_duration += other.indexing_duration();
        self.desolved_duration += other.desolved_duration();
        self.basic_solve_duration += other.basic_solve_duration();
        self.partition_duration += other.partition_duration();
    }

    pub(super) fn solve_passes(&self) -> usize {
        self.solve_passes
    }
    pub(super) fn inc_solve_passes(&mut self) {
        self.solve_passes += 1;
    }

    pub(super) fn max_partitions(&self) -> usize {
        self.max_partitions
    }
    pub(super) fn set_max_partitions(&mut self, maybe_max: usize) {
        self.max_partitions = cmp::max(self.max_partitions, maybe_max);
    }

    pub(super) fn max_partition_size(&self) -> usize {
        self.max_partition_size
    }
    pub(super) fn set_max_partition_size(&mut self, maybe_max: usize) {
        self.max_partition_size = cmp::max(self.max_partition_size, maybe_max);
    }

    pub(super) fn partitions_created(&self) -> usize {
        self.partitions_created
    }
    pub(super) fn inc_partitions_created(&mut self) {
        self.partitions_created += 1;
    }

    pub(super) fn neighbor_visits(&self) -> usize {
        self.neighbor_visits
    }
    pub(super) fn inc_neighbor_visits(&mut self, directions: DirectionSet) {
        self.neighbor_visits += directions.len() as usize;
    }

    pub(super) fn indexing_duration(&self) -> Duration {
        self.indexing_duration
    }
    pub(super) fn inc_indexing_duration(&mut self) {
        let now = Instant::now();
        self.indexing_duration += now - *self.mark_time.get_or_insert(Instant::now());
        self.mark_time = Some(now);
    }

    pub(super) fn desolved_duration(&self) -> Duration {
        self.desolved_duration
    }
    pub(super) fn inc_desolved_duration(&mut self) {
        let now = Instant::now();
        self.desolved_duration += now - *self.mark_time.get_or_insert(Instant::now());
        self.mark_time = Some(now);
    }

    pub(super) fn basic_solve_duration(&self) -> Duration {
        self.basic_solve_duration
    }
    pub(super) fn inc_basic_solve_duration(&mut self) {
        let now = Instant::now();
        self.basic_solve_duration += now - *self.mark_time.get_or_insert(Instant::now());
        self.mark_time = Some(now);
    }

    pub(super) fn partition_duration(&self) -> Duration {
        self.partition_duration
    }
    pub(super) fn inc_partition_duration(&mut self) {
        let now = Instant::now();
        self.partition_duration += now - *self.mark_time.get_or_insert(Instant::now());
        self.mark_time = Some(now);
    }

    pub(super) fn mark_now(&mut self) {
        self.mark_time = Some(Instant::now());
    }

    pub(super) fn total_duration(&self) -> Duration {
        self.desolved_duration()
            + self.indexing_duration()
            + self.partition_duration()
            + self.basic_solve_duration()
    }
}

#[cfg(any(debug_assertions, test))]
impl Display for Metrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let games = self.games_played().max(1) as f64;
        let solves = self.solve_passes().max(1) as f64;

        pub(super) fn write_usize_field(
            f: &mut fmt::Formatter<'_>,
            text: &str,
            val: usize,
            games: f64,
            solves: f64,
        ) -> fmt::Result {
            write!(f, "\n    {text}: {val}")?;
            if games > 1.0 || solves > 1.0 {
                write!(f, " (")?;
                if games > 1.0 {
                    write!(f, "{:.2} per game", val as f64 / games)?;
                    if solves > 1.0 {
                        write!(f, ", ")?;
                    }
                }
                if solves > 1.0 {
                    write!(f, "{:.2} per solve", val as f64 / solves)?;
                }
                write!(f, ")")?;
            }
            write!(f, ",")
        }
        pub(super) fn write_duration_field(
            f: &mut fmt::Formatter<'_>,
            text: &str,
            val: Duration,
            games: f64,
            solves: f64,
        ) -> fmt::Result {
            write!(f, "\n    {text}: {val:?}")?;
            if games > 1.0 || solves > 1.0 {
                write!(f, " (")?;
                if games > 1.0 {
                    write!(f, "{:?} per game", val / games as u32)?;
                    if solves > 1.0 {
                        write!(f, ", ")?;
                    }
                }
                if solves > 1.0 {
                    write!(f, "{:?} per solve", val / solves as u32)?;
                }
                write!(f, ")")?;
            }
            write!(f, ",")
        }
        write!(f, "Metrics {{")?;
        write_usize_field(f, "games", self.games_played(), 0.0, 0.0)?;
        write_usize_field(f, "solves", self.solve_passes(), games, 0.0)?;
        write_usize_field(
            f,
            "Max simultaneous partitions",
            self.max_partitions(),
            0.0,
            0.0,
        )?;
        write_usize_field(f, "Max partition size", self.max_partition_size(), 0.0, 0.0)?;
        write_usize_field(
            f,
            "Partitions created",
            self.partitions_created(),
            games,
            solves,
        )?;
        write_usize_field(f, "Neighbor visits", self.neighbor_visits(), games, solves)?;
        write_duration_field(
            f,
            "Time spent on indexing",
            self.indexing_duration(),
            games,
            solves,
        )?;
        write_duration_field(
            f,
            "Time spent on resetting prior inferences",
            self.desolved_duration(),
            games,
            solves,
        )?;
        write_duration_field(
            f,
            "Time spent on basic solve inferences",
            self.basic_solve_duration(),
            games,
            solves,
        )?;
        write_duration_field(
            f,
            "Time spent on partition",
            self.partition_duration(),
            games,
            solves,
        )?;
        write_duration_field(f, "Total time", self.total_duration(), games, solves)?;

        writeln!(f, "\n}}")?;
        Ok(())
    }
}

#[cfg(not(any(debug_assertions, test)))]
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Metrics;

#[cfg(not(any(debug_assertions, test)))]
impl Metrics {
    #[inline(always)]
    pub(super) fn inc_solve_passes(&mut self) {}

    #[inline(always)]
    pub(super) fn set_max_partitions(&mut self, _maybe_max: usize) {}

    #[inline(always)]
    pub(super) fn set_max_partition_size(&mut self, _maybe_max: usize) {}

    #[inline(always)]
    pub(super) fn inc_partitions_created(&mut self) {}

    #[inline(always)]
    pub(super) fn inc_neighbor_visits(&mut self, _directions: DirectionSet) {}

    #[inline(always)]
    pub(super) fn inc_indexing_duration(&mut self) {}

    #[inline(always)]
    pub(super) fn inc_desolved_duration(&mut self) {}

    #[inline(always)]
    pub(super) fn inc_basic_solve_duration(&mut self) {}

    #[inline(always)]
    pub(super) fn inc_partition_duration(&mut self) {}

    #[inline(always)]
    pub(super) fn mark_now(&mut self) {}
}

#[cfg(not(any(debug_assertions, test)))]
impl Display for Metrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Metrics not available in release mode")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::grid::{E, N, W};

    #[test]
    fn test_metrics_defaults_and_increments() {
        let mut m = Metrics::default();
        assert_eq!(m.games_played(), 0);
        assert_eq!(m.solve_passes(), 0);
        assert_eq!(m.max_partitions(), 0);
        assert_eq!(m.max_partition_size(), 0);
        assert_eq!(m.partitions_created(), 0);
        assert_eq!(m.neighbor_visits(), 0);
        assert_eq!(m.total_duration(), Duration::ZERO);

        m.inc_solve_passes();
        assert_eq!(m.solve_passes(), 1);

        m.inc_partitions_created();
        assert_eq!(m.partitions_created(), 1);

        m.set_max_partitions(3);
        m.set_max_partitions(2);
        assert_eq!(m.max_partitions(), 3);

        m.set_max_partition_size(5);
        m.set_max_partition_size(4);
        assert_eq!(m.max_partition_size(), 5);

        m.inc_neighbor_visits(N | E | W);
        assert_eq!(m.neighbor_visits(), 3);
    }

    #[test]
    fn test_metrics_durations_and_display() {
        let mut m = Metrics::default();
        m.mark_now();
        m.inc_indexing_duration();
        m.inc_desolved_duration();
        m.inc_basic_solve_duration();
        m.inc_partition_duration();

        let total = m.total_duration();
        assert_eq!(
            total,
            m.indexing_duration()
                + m.desolved_duration()
                + m.basic_solve_duration()
                + m.partition_duration()
        );

        let disp = format!("{}", m);
        assert!(disp.contains("Metrics {"));
        assert!(disp.contains("games: 0"));
    }

    #[test]
    fn test_metrics_tally_game() {
        let mut game1 = Metrics::default();
        game1.inc_solve_passes();
        game1.inc_partitions_created();
        game1.set_max_partitions(2);
        game1.set_max_partition_size(4);
        game1.inc_neighbor_visits(N | E);

        let mut cumulative = Metrics::default();
        cumulative.tally_game(&game1);
        assert_eq!(cumulative.games_played(), 1);
        assert_eq!(cumulative.solve_passes(), 1);
        assert_eq!(cumulative.partitions_created(), 1);
        assert_eq!(cumulative.max_partitions(), 2);
        assert_eq!(cumulative.max_partition_size(), 4);
        assert_eq!(cumulative.neighbor_visits(), 2);

        let mut game2 = Metrics::default();
        game2.inc_solve_passes();
        game2.set_max_partitions(5);
        game2.set_max_partition_size(8);

        cumulative.tally_game(&game2);
        assert_eq!(cumulative.games_played(), 2);
        assert_eq!(cumulative.solve_passes(), 2);
        assert_eq!(cumulative.max_partitions(), 5);
        assert_eq!(cumulative.max_partition_size(), 8);

        let disp = format!("{}", cumulative);
        assert!(disp.contains("per game"));
    }
}

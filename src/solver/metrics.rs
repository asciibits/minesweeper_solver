//! Instrumentation and phase-timing counters collected during debug/test builds.
//!
//! In release builds without the `debug` feature, [`Metrics`] compiles down to a
//! zero-sized no-op struct so instrumentation has zero runtime cost.

#[cfg(any(debug_assertions, test, feature = "debug"))]
use std::cmp;
use std::fmt;
use std::fmt::Display;
#[cfg(any(debug_assertions, test, feature = "debug"))]
use std::time::{Duration, Instant};

use crate::solver::grid::DirectionSet;

#[cfg(any(debug_assertions, test, feature = "debug"))]
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Metrics {
    solve_passes: usize,
    max_partitions: usize,
    max_partition_size: u32,
    partitions_created: usize,
    neighbor_visits: usize,
    last_indexing_duration: u64,
    indexing_duration: AggregatedMetric,
    last_desolved_duration: u64,
    desolved_duration: AggregatedMetric,
    last_solve_duration: u64,
    solve_duration: AggregatedMetric,
    partition_duration: AggregatedMetric,
    total_time_duration: AggregatedMetric,
    mark_time: Option<Instant>,
}

#[cfg(any(debug_assertions, test, feature = "debug"))]
impl Metrics {
    #[cfg(any(test, feature = "debug"))]
    pub(super) fn solve_passes(&self) -> usize {
        self.solve_passes
    }
    pub(super) fn inc_solve_passes(&mut self) {
        self.solve_passes += 1;
    }

    #[cfg(test)]
    pub(super) fn max_partitions(&self) -> usize {
        self.max_partitions
    }
    pub(super) fn set_max_partitions(&mut self, maybe_max: usize) {
        self.max_partitions = cmp::max(self.max_partitions, maybe_max);
    }

    #[cfg(test)]
    pub(super) fn max_partition_size(&self) -> u32 {
        self.max_partition_size
    }
    pub(super) fn set_max_partition_size(&mut self, maybe_max: u32) {
        self.max_partition_size = cmp::max(self.max_partition_size, maybe_max);
    }

    #[cfg(test)]
    pub(super) fn partitions_created(&self) -> usize {
        self.partitions_created
    }
    pub(super) fn inc_partitions_created(&mut self) {
        self.partitions_created += 1;
    }

    #[cfg(test)]
    pub(super) fn neighbor_visits(&self) -> usize {
        self.neighbor_visits
    }
    pub(super) fn inc_neighbor_visits(&mut self, directions: DirectionSet) {
        self.neighbor_visits += directions.len() as usize;
    }

    pub(super) fn indexing_duration(&self) -> AggregatedMetric {
        self.indexing_duration
    }
    pub(super) fn inc_indexing_duration(&mut self) {
        let now = Instant::now();
        let duration = now - *self.mark_time.get_or_insert(now);
        self.last_indexing_duration = duration.as_nanos() as u64;
        self.indexing_duration.apply(duration.as_nanos() as f64);
        self.mark_time = Some(now);
    }

    pub(super) fn desolved_duration(&self) -> AggregatedMetric {
        self.desolved_duration
    }
    pub(super) fn inc_desolved_duration(&mut self) {
        let now = Instant::now();
        let duration = now - *self.mark_time.get_or_insert(now);
        self.last_desolved_duration = duration.as_nanos() as u64;
        self.desolved_duration.apply(duration.as_nanos() as f64);
        self.mark_time = Some(now);
    }

    pub(super) fn basic_solve_duration(&self) -> AggregatedMetric {
        self.solve_duration
    }
    pub(super) fn inc_basic_solve_duration(&mut self) {
        let now = Instant::now();
        let duration = now - *self.mark_time.get_or_insert(now);
        self.last_solve_duration = duration.as_nanos() as u64;
        self.solve_duration.apply(duration.as_nanos() as f64);
        self.mark_time = Some(now);
    }

    pub(super) fn partition_duration(&self) -> AggregatedMetric {
        self.partition_duration
    }
    pub(super) fn inc_partition_duration(&mut self) {
        let now = Instant::now();
        let duration = now - *self.mark_time.get_or_insert(now);
        self.partition_duration.apply(duration.as_nanos() as f64);
        self.total_time_duration.apply(
            (self.last_desolved_duration
                + self.last_indexing_duration
                + self.last_solve_duration
                + duration.as_nanos() as u64) as f64,
        );
        self.mark_time = Some(now);
    }

    pub(super) fn mark_now(&mut self) {
        self.mark_time = Some(Instant::now());
    }

    pub(super) fn total_duration(&self) -> AggregatedMetric {
        self.total_time_duration
    }
}

#[cfg(any(debug_assertions, test, feature = "debug"))]
impl Display for Metrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        pub(super) fn write_duration_field(
            f: &mut fmt::Formatter<'_>,
            text: &str,
            val: AggregatedMetric,
        ) -> fmt::Result {
            write!(
                f,
                "\n    {text}: {:?}",
                Duration::from_nanos(val.total() as u64)
            )?;
            if val.count() > 1 {
                write!(
                    f,
                    "({:?} +/- {:?} per solve - min/max: {:?}/{:?})",
                    Duration::from_nanos(val.mean() as u64),
                    Duration::from_nanos(val.stdev() as u64),
                    Duration::from_nanos(val.min as u64),
                    Duration::from_nanos(val.max as u64),
                )?;
            }
            write!(f, ",")
        }
        write!(f, "Metrics {{")?;
        write!(f, "\n    solves: {}", self.solve_passes)?;
        write!(
            f,
            "\n    Max simultaneous partitions: {}",
            self.max_partitions
        )?;
        write!(f, "\n    Max partition size: {}", self.max_partition_size)?;
        write!(
            f,
            "\n    Partitions created {} ({:.3} per solve)",
            self.partitions_created,
            self.partitions_created as f64 / self.solve_passes as f64
        )?;
        write!(
            f,
            "\n    Neighbor visits {} ({:.3} per solve)",
            self.neighbor_visits,
            self.neighbor_visits as f64 / self.solve_passes as f64
        )?;
        write_duration_field(f, "Time spent on indexing", self.indexing_duration())?;
        write_duration_field(
            f,
            "Time spent on resetting prior inferences",
            self.desolved_duration(),
        )?;
        write_duration_field(
            f,
            "Time spent on basic solve inferences",
            self.basic_solve_duration(),
        )?;
        write_duration_field(f, "Time spent on partition", self.partition_duration())?;
        write_duration_field(f, "Total time", self.total_duration())?;

        writeln!(f, "\n}}")?;
        Ok(())
    }
}

#[cfg(test)]
#[derive(Debug, Default, Copy, Clone)]
pub(super) struct AggregatedMetrics {
    solve_passes: AggregatedMetric,
    max_partitions: usize,
    max_partition_size: u32,
    partitions_created: AggregatedMetric,
    neighbor_visits: AggregatedMetric,
    indexing_duration: AggregatedMetric,
    desolved_duration: AggregatedMetric,
    solve_duration: AggregatedMetric,
    partition_duration: AggregatedMetric,
    total_duration: AggregatedMetric,
}

#[cfg(test)]
impl AggregatedMetrics {
    pub(super) fn apply_metrics(&mut self, metrics: &Metrics) {
        self.solve_passes.apply(metrics.solve_passes as f64);
        self.max_partitions = self.max_partitions.max(metrics.max_partitions);
        self.max_partition_size = self.max_partition_size.max(metrics.max_partition_size);
        self.partitions_created
            .apply(metrics.partitions_created as f64);
        self.neighbor_visits.apply(metrics.neighbor_visits as f64);
        self.indexing_duration
            .apply(metrics.indexing_duration.total());
        self.desolved_duration
            .apply(metrics.desolved_duration.total());
        self.solve_duration.apply(metrics.solve_duration.total());
        self.partition_duration
            .apply(metrics.partition_duration.total());
        self.total_duration
            .apply(metrics.total_time_duration.total());
    }

    pub(super) fn is_empty(&self) -> bool {
        self.solve_passes.n == 0.0
    }
}

#[cfg(test)]
impl Display for AggregatedMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        pub(super) fn write_usize_field(
            f: &mut fmt::Formatter<'_>,
            text: &str,
            val: AggregatedMetric,
        ) -> fmt::Result {
            let total = val.total();
            if total.fract() == 0.0 {
                write!(f, "\n    {text}: {}", total as u64)?;
            } else {
                write!(f, "\n    {text}: {:.3}", total)?;
            }
            if val.count() > 1 {
                write!(
                    f,
                    "({:.3} +/- {:.3} per game, min/max: {}/{})",
                    val.mean(),
                    val.stdev(),
                    val.min,
                    val.max
                )?;
            }
            write!(f, ",")
        }
        pub(super) fn write_duration_field(
            f: &mut fmt::Formatter<'_>,
            text: &str,
            val: AggregatedMetric,
        ) -> fmt::Result {
            write!(
                f,
                "\n    {text}: {:?}",
                Duration::from_nanos(val.total() as u64)
            )?;
            if val.count() > 1 {
                write!(
                    f,
                    "({:?} +/- {:?} per game, min/max: {:?}/{:?})",
                    Duration::from_nanos(val.mean() as u64),
                    Duration::from_nanos(val.stdev() as u64),
                    Duration::from_nanos(val.min as u64),
                    Duration::from_nanos(val.max as u64),
                )?;
            }
            write!(f, ",")
        }
        write!(f, "Metrics {{")?;
        write_usize_field(f, "solves", self.solve_passes)?;
        write!(
            f,
            "\n    Max simultaneous partitions: {}",
            self.max_partitions
        )?;
        write!(f, "\n    Max partition size: {}", self.max_partition_size)?;
        write_usize_field(f, "Partitions created", self.partitions_created)?;
        write_usize_field(f, "Neighbor visits", self.neighbor_visits)?;
        write_duration_field(f, "Time spent on indexing", self.indexing_duration)?;
        write_duration_field(
            f,
            "Time spent on resetting prior inferences",
            self.desolved_duration,
        )?;
        write_duration_field(
            f,
            "Time spent on basic solve inferences",
            self.solve_duration,
        )?;
        write_duration_field(f, "Time spent on partition", self.partition_duration)?;
        write_duration_field(f, "Total time", self.total_duration)?;

        writeln!(f, "\n}}")?;
        Ok(())
    }
}

#[cfg(any(debug_assertions, test, feature = "debug"))]
#[derive(Debug, Default, Copy, Clone)]
pub(super) struct AggregatedMetric {
    n: f64,
    total: f64,
    mean: f64,
    s: f64,
    max: f64,
    min: f64,
}

#[cfg(any(debug_assertions, test, feature = "debug"))]
impl AggregatedMetric {
    fn apply(&mut self, metric: f64) {
        // Welford’s Algorithm
        self.max = if self.n == 0.0 {
            metric
        } else {
            metric.max(self.max)
        };
        self.min = if self.n == 0.0 {
            metric
        } else {
            metric.min(self.min)
        };
        self.n += 1.0;
        self.total += metric;
        let mean = self.mean + (metric - self.mean) / self.n;
        self.s += (metric - mean) * (metric - self.mean);
        self.mean = mean;
    }

    fn count(&self) -> u64 {
        self.n as u64
    }

    fn total(&self) -> f64 {
        self.total
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn stdev(&self) -> f64 {
        if self.n > 1.0 {
            (self.s / (self.n - 1.0)).sqrt()
        } else {
            0.0
        }
    }
}

#[cfg(not(any(debug_assertions, test, feature = "debug")))]
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Metrics;

#[cfg(not(any(debug_assertions, test, feature = "debug")))]
impl Metrics {
    #[inline(always)]
    pub(super) fn inc_solve_passes(&mut self) {}

    #[inline(always)]
    pub(super) fn set_max_partitions(&mut self, _maybe_max: usize) {}

    #[inline(always)]
    pub(super) fn set_max_partition_size(&mut self, _maybe_max: u32) {}

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

#[cfg(not(any(debug_assertions, test, feature = "debug")))]
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
        assert_eq!(m.solve_passes(), 0);
        assert_eq!(m.max_partitions(), 0);
        assert_eq!(m.max_partition_size(), 0);
        assert_eq!(m.partitions_created(), 0);
        assert_eq!(m.neighbor_visits(), 0);

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
        // Zero solve passes should not panic when displayed
        let zero_disp = format!("{}", m);
        assert!(zero_disp.contains("Metrics {"));
        assert!(zero_disp.contains("solves: 0"));

        m.inc_solve_passes();
        m.inc_partitions_created();
        m.mark_now();
        m.inc_indexing_duration();
        m.inc_desolved_duration();
        m.inc_basic_solve_duration();
        m.inc_partition_duration();

        let total = m.total_duration().total();
        let sum = m.indexing_duration().total()
            + m.desolved_duration().total()
            + m.basic_solve_duration().total()
            + m.partition_duration().total();
        assert!((total - sum).abs() < 1e-6);

        let disp = format!("{}", m);
        assert!(disp.contains("Metrics {"));
        assert!(disp.contains("solves: 1"));
        assert!(disp.contains("\n    Partitions created 1 (1.000 per solve)"));
        assert!(disp.contains("\n    Neighbor visits 0 (0.000 per solve)"));
        assert!(disp.contains("Total time:"));
    }

    #[test]
    fn test_aggregated_metrics() {
        let mut agg = AggregatedMetrics::default();
        assert!(agg.is_empty());

        let mut m1 = Metrics::default();
        m1.inc_solve_passes();
        m1.inc_partitions_created();
        m1.set_max_partitions(2);
        m1.set_max_partition_size(4);
        m1.inc_neighbor_visits(N | E);
        agg.apply_metrics(&m1);
        assert!(!agg.is_empty());

        let mut m2 = Metrics::default();
        m2.inc_solve_passes();
        m2.inc_solve_passes();
        m2.set_max_partitions(5);
        m2.set_max_partition_size(3);
        m2.inc_neighbor_visits(!!W);
        agg.apply_metrics(&m2);

        let disp = format!("{agg}");
        assert!(disp.contains("Metrics {"));
        assert!(disp.contains("solves: 3(1.500 +/- "));
        assert!(disp.contains("Max simultaneous partitions: 5"));
        assert!(disp.contains("Max partition size: 4"));
        assert!(disp.contains("Partitions created: 1(0.500 +/- "));
        assert!(disp.contains("Neighbor visits: 3(1.500 +/- "));
    }
}

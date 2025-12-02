// Phase 3: Text Removal (Pass-through)
//
// Text removal now happens in Phase 1 using the FPN text cleaner.
// This phase simply passes through the already-cleaned regions.

use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, instrument};

use crate::core::config::Config;
use crate::core::types::{ImageData, Phase1Output, Phase3Output};

/// Phase 3 pipeline: Pass-through of cleaned regions
///
/// Text cleaning is now integrated into Phase 1 using the FPN text detector.
/// This phase simply extracts the pre-cleaned regions from Phase1Output.
pub struct Phase3Pipeline {
    #[allow(dead_code)]
    config: Arc<Config>,
}

impl Phase3Pipeline {
    pub fn new(config: Arc<Config>) -> Self {
        Self { config }
    }

    /// Execute Phase 3: Pass through pre-cleaned regions from Phase 1
    ///
    /// Since text cleaning now happens in Phase 1, this is just a pass-through.
    /// Banana-processed regions are filtered out (they have their own images).
    #[instrument(skip(self, _image_data, phase1_output, banana_processed_ids), fields(
        page = phase1_output.page_index,
        regions = phase1_output.regions.len()
    ))]
    pub async fn execute(
        &self,
        _image_data: &ImageData,
        phase1_output: &Phase1Output,
        banana_processed_ids: &[usize],
        _blur_free_text: bool,
        _use_mask: bool,
    ) -> Result<Phase3Output> {
        let start = std::time::Instant::now();

        // Filter out banana-processed regions (they have their own cleaned images)
        let cleaned_regions: Vec<(usize, Vec<u8>, [i32; 4])> = phase1_output.cleaned_regions
            .iter()
            .filter(|(region_id, _, _)| !banana_processed_ids.contains(region_id))
            .cloned()
            .collect();

        debug!(
            "Phase 3 pass-through: {} regions in {:.0}ms",
            cleaned_regions.len(),
            start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(Phase3Output {
            page_index: phase1_output.page_index,
            cleaned_regions,
        })
    }
}

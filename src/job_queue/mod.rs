// Async job queue for background processing of translation batches

use crate::pipeline::ProcessingPipeline;
use crate::types::{BatchConfig, BatchResult, ImageData};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::Instant;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Job status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

/// Job metadata
#[derive(Debug, Clone)]
pub struct Job {
    pub id: Uuid,
    pub status: JobStatus,
    pub progress: f64,
    pub created_at: Instant,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub error: Option<String>,
    pub result: Option<BatchResult>,
}

impl Job {
    fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            status: JobStatus::Pending,
            progress: 0.0,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            error: None,
            result: None,
        }
    }

    pub fn processing_time_ms(&self) -> Option<f64> {
        if let (Some(started), Some(completed)) = (self.started_at, self.completed_at) {
            Some((completed - started).as_secs_f64() * 1000.0)
        } else {
            None
        }
    }
}

/// Job queue for async processing
pub struct JobQueue {
    jobs: Arc<RwLock<HashMap<Uuid, Job>>>,
    pipeline: Arc<ProcessingPipeline>,
}

impl JobQueue {
    pub fn new(pipeline: Arc<ProcessingPipeline>) -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            pipeline,
        }
    }

    /// Submit a new job
    pub async fn submit(
        &self,
        images: Vec<ImageData>,
        config: BatchConfig,
    ) -> Uuid {
        let job = Job::new();
        let job_id = job.id;

        debug!("Submitting job {} with {} images", job_id, images.len());

        // Store job
        self.jobs.write().await.insert(job_id, job.clone());

        // Spawn background processing
        let jobs = Arc::clone(&self.jobs);
        let pipeline = Arc::clone(&self.pipeline);

        tokio::spawn(async move {
            Self::process_job(job_id, images, config, jobs, pipeline).await;
        });

        job_id
    }

    /// Process a job in the background
    async fn process_job(
        job_id: Uuid,
        images: Vec<ImageData>,
        config: BatchConfig,
        jobs: Arc<RwLock<HashMap<Uuid, Job>>>,
        pipeline: Arc<ProcessingPipeline>,
    ) {
        info!("Starting job {}", job_id);

        // Update status to processing
        {
            let mut jobs_map = jobs.write().await;
            if let Some(job) = jobs_map.get_mut(&job_id) {
                job.status = JobStatus::Processing;
                job.started_at = Some(Instant::now());
            }
        }

        // Process the batch
        let result = pipeline.process_batch(images, &config).await;

        // Update job with result
        let mut jobs_map = jobs.write().await;
        if let Some(job) = jobs_map.get_mut(&job_id) {
            job.completed_at = Some(Instant::now());

            match result {
                Ok((results, metrics)) => {
                    info!("Job {} completed successfully", job_id);
                    job.status = JobStatus::Completed;
                    job.progress = 1.0;
                    job.result = Some(BatchResult {
                        total: results.len(),
                        successful: results.iter().filter(|r| r.success).count(),
                        failed: results.iter().filter(|r| !r.success).count(),
                        processing_time_ms: job.processing_time_ms().unwrap_or(0.0),
                        average_time_per_page_ms: job.processing_time_ms().unwrap_or(0.0) / results.len() as f64,
                        analytics: crate::types::BatchAnalytics {
                            total_bubbles_detected: results.iter().map(|r| r.bubbles_detected).sum(),
                            total_bubbles_translated: results.iter().map(|r| r.bubbles_translated).sum(),
                            cache_hit_rate: if metrics.cache_hits + metrics.cache_misses > 0 {
                                metrics.cache_hits as f64 / (metrics.cache_hits + metrics.cache_misses) as f64 * 100.0
                            } else {
                                0.0
                            },
                            cache_hits: metrics.cache_hits,
                            cache_misses: metrics.cache_misses,
                            api_calls_made: metrics.api_calls_made,
                            api_calls_saved: metrics.api_calls_saved,
                            background_redraws: metrics.background_redraws,
                            simple_backgrounds: metrics.simple_backgrounds,
                            average_detection_time_ms: 0.0,
                            average_translation_time_ms: 0.0,
                            average_rendering_time_ms: 0.0,
                            model_used: crate::types::ModelInfo {
                                translation_model: config.translation_model.unwrap_or_default(),
                                image_gen_model: config.image_gen_model.unwrap_or_default(),
                            },
                        },
                        results,
                    });
                }
                Err(e) => {
                    warn!("Job {} failed: {}", job_id, e);
                    job.status = JobStatus::Failed;
                    job.error = Some(e.to_string());
                }
            }
        }
    }

    /// Get job status
    pub async fn get_job(&self, job_id: Uuid) -> Option<Job> {
        self.jobs.read().await.get(&job_id).cloned()
    }

    /// Get job result
    pub async fn get_result(&self, job_id: Uuid) -> Option<BatchResult> {
        self.jobs
            .read()
            .await
            .get(&job_id)
            .and_then(|job| job.result.clone())
    }

    /// Cancel a job (if still pending)
    pub async fn cancel(&self, job_id: Uuid) -> bool {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            if job.status == JobStatus::Pending {
                job.status = JobStatus::Failed;
                job.error = Some("Job cancelled by user".to_string());
                return true;
            }
        }
        false
    }

    /// Clean up old completed jobs (older than TTL)
    pub async fn cleanup(&self, ttl_seconds: u64) {
        let mut jobs = self.jobs.write().await;
        let now = Instant::now();
        let ttl = std::time::Duration::from_secs(ttl_seconds);

        jobs.retain(|id, job| {
            let should_keep = match job.status {
                JobStatus::Completed | JobStatus::Failed => {
                    if let Some(completed_at) = job.completed_at {
                        now.duration_since(completed_at) < ttl
                    } else {
                        true
                    }
                }
                _ => true,
            };

            if !should_keep {
                debug!("Cleaning up old job {}", id);
            }

            should_keep
        });
    }

    /// Get statistics
    pub async fn stats(&self) -> (usize, usize, usize, usize) {
        let jobs = self.jobs.read().await;
        let pending = jobs.values().filter(|j| j.status == JobStatus::Pending).count();
        let processing = jobs.values().filter(|j| j.status == JobStatus::Processing).count();
        let completed = jobs.values().filter(|j| j.status == JobStatus::Completed).count();
        let failed = jobs.values().filter(|j| j.status == JobStatus::Failed).count();
        (pending, processing, completed, failed)
    }
}

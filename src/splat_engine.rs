use crate::adapter::SplatAdapter;
use crate::llm::qwen::Model as MyBaseModel;
use crate::token_promotion::dynamic_tokenizer::DynamicTokenizer;
use candle_core::{Device, IndexOp, Result, Tensor};

pub struct SplatEngine {
    pub base_model: MyBaseModel, // Retained for compatibility; unused in the current steering path
    pub adapter: SplatAdapter,   // Retained for compatibility
    pub tokenizer: DynamicTokenizer, // Used for decoding
    pub device: Device,
    pub steering_vectors: Vec<Tensor>,
    pub counter_steering_vectors: Vec<Tensor>,
    pub gain: f64,
}

impl SplatEngine {
    /// Initialize the Engine
    pub fn new(
        base_model: MyBaseModel,
        adapter: SplatAdapter,
        tokenizer: DynamicTokenizer,
        device: Device,
    ) -> Self {
        Self {
            base_model,
            adapter,
            tokenizer,
            device,
            steering_vectors: Vec::new(),
            counter_steering_vectors: Vec::new(),
            gain: 0.6,
        }
    }

    pub fn set_gain(&mut self, gain: f64) {
        self.gain = gain;
    }

    pub fn get_gain(&self) -> f64 {
        self.gain
    }

    pub fn clear_steering_vectors(&mut self) {
        self.steering_vectors.clear();
        self.counter_steering_vectors.clear();
    }

    pub fn add_steering_vector(&mut self, vector: Tensor) {
        self.steering_vectors.push(vector);
    }

    pub fn add_counter_steering_vector(&mut self, vector: Tensor) {
        self.counter_steering_vectors.push(vector);
    }

    pub fn inject_steering_sequence(&mut self, sequence: &Tensor) -> Result<()> {
        Ok(())
    }

    /// Physics-based generation step for the experimental splat pathway.
    pub fn step(&mut self, input_tensor: &Tensor) -> anyhow::Result<Tensor> {
        // 1. Decode Input to Text (Reverse Tokenization)
        let input_ids: Vec<u32> = input_tensor.flatten_all()?.to_vec1()?;
        let _text = self
            .tokenizer
            .decode_batch(&[input_ids.clone()], true)
            .map(|v| v.first().cloned().unwrap_or_default())
            .unwrap_or_default();

        // 2. Run the experimental steering stage
        // We need access to the memory store.
        // SplatEngine doesn't own the store directly in this architecture (MemorySystem does).
        // For now, return a placeholder control token that asks the outer loop
        // to invoke the physics path.

        // Since we can't easily access the store here without refactoring `main.rs` to pass it in,
        // The outer loop interprets this token as a request to enter the
        // experimental physics path.

        // Token ID 999999 = placeholder control token
        let physics_token = 999999u32;

        // Return a tensor with just this token
        let device = input_tensor.device();
        let out = Tensor::new(&[physics_token], device)?.unsqueeze(0)?;

        Ok(out)
    }

    pub fn step_with_injection(
        &mut self,
        input_ids: &Tensor,
        _embedding: Vec<f32>,
    ) -> Result<Tensor> {
        self.step(input_ids)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

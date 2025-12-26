/**
 * Z-Image-Turbo Frontend Application
 * Alpine.js-based reactive UI for AI image and video generation
 */

function app() {
    return {
        // Tab state
        activeTab: 'image',

        // Image generation state
        imagePrompt: '',
        currentImage: null,
        isGenerating: false,
        isEnhancing: false,
        progress: 0,
        progressText: 'Starting...',
        statusMessage: '',
        statusType: 'success',
        gallery: [],

        // Image settings
        imageSettings: {
            aspectRatio: '1:1',
            seed: -1,
            steps: 8,
            numImages: 1
        },

        // Video generation state
        videoPrompt: '',
        currentVideo: null,
        isGeneratingVideo: false,
        isEnhancingVideo: false,
        videoProgress: 0,
        videoProgressText: 'Starting...',
        videoStatusMessage: '',
        videoStatusType: 'success',

        // Video duration (in seconds) - frames calculated as duration * 16 + 1
        videoDuration: 5,

        // Video settings (I2V only)
        videoSettings: {
            resolution: '480p',
            seed: -1,
            frames: 81,  // Will be calculated from videoDuration
            modelHigh: '',
            modelLow: '',
            loraHigh: '',
            loraLow: '',
            inputImage: null,
            inputImagePreview: null,
            upscaleEnabled: false,
            upscaleModel: '',
            interpolateEnabled: false,
            interpolateModel: '',
            interpolateMultiplier: 2
        },

        // UI state
        showVideoModels: true,
        showEnhancement: false,
        isDragging: false,

        // Models data
        models: {
            video: { high: [], low: [], loras_high: [], loras_low: [] },
            enhancement: { upscale: [], vfi: [] },
            image: { aspect_ratios: [] }
        },

        // Toast notifications
        toasts: [],
        toastId: 0,

        // WebSocket connection
        ws: null,

        /**
         * Initialize the application
         */
        async init() {
            console.log('Z-Image-Turbo initializing...');

            // Calculate initial frames from duration
            this.videoSettings.frames = this.videoDuration * 16 + 1;

            // Watch videoDuration changes and update frames
            this.$watch('videoDuration', (value) => {
                this.videoSettings.frames = value * 16 + 1;
                console.log(`Duration: ${value}s → Frames: ${this.videoSettings.frames}`);
            });

            // Load models
            await this.refreshModels();

            // Load gallery from server
            await this.loadGallery();

            console.log('Ready!');
        },

        /**
         * Show a toast notification
         */
        showToast(message, type = 'success', duration = 3000) {
            const id = ++this.toastId;
            const toast = { id, message, type, visible: true };
            this.toasts.push(toast);

            setTimeout(() => {
                toast.visible = false;
                setTimeout(() => {
                    this.toasts = this.toasts.filter(t => t.id !== id);
                }, 300);
            }, duration);
        },

        /**
         * Refresh available models from API
         */
        async refreshModels() {
            try {
                const response = await fetch('/api/models');
                if (response.ok) {
                    const data = await response.json();

                    // Ensure all arrays exist
                    this.models = {
                        video: {
                            high: data.video?.high || [],
                            low: data.video?.low || [],
                            loras_high: data.video?.loras_high || [],
                            loras_low: data.video?.loras_low || []
                        },
                        enhancement: {
                            upscale: data.enhancement?.upscale || [],
                            vfi: data.enhancement?.vfi || []
                        },
                        image: data.image || { aspect_ratios: [] }
                    };

                    console.log('Models loaded:', this.models);
                    console.log('Upscale models:', this.models.enhancement.upscale);
                    console.log('VFI models:', this.models.enhancement.vfi);

                    // Set default selections (always update when refreshing)
                    if (this.models.video.high.length > 0) {
                        this.videoSettings.modelHigh = this.models.video.high[0];
                    }
                    if (this.models.video.low.length > 0) {
                        this.videoSettings.modelLow = this.models.video.low[0];
                    }
                    if (this.models.video.loras_high.length > 0) {
                        this.videoSettings.loraHigh = this.models.video.loras_high[0];
                    }
                    if (this.models.video.loras_low.length > 0) {
                        this.videoSettings.loraLow = this.models.video.loras_low[0];
                    }
                    if (this.models.enhancement.upscale.length > 0) {
                        this.videoSettings.upscaleModel = this.models.enhancement.upscale[0];
                    }
                    if (this.models.enhancement.vfi.length > 0) {
                        this.videoSettings.interpolateModel = this.models.enhancement.vfi[0];
                    }

                    this.showToast(`Models refreshed (${this.models.enhancement.upscale.length} upscale, ${this.models.enhancement.vfi.length} VFI)`);
                }
            } catch (error) {
                console.error('Failed to load models:', error);
                this.showToast('Failed to load models', 'error');
            }
        },

        /**
         * Load gallery from server
         */
        async loadGallery() {
            try {
                const response = await fetch('/api/gallery');
                if (response.ok) {
                    const data = await response.json();
                    this.gallery = data.images || [];
                }
            } catch (error) {
                console.error('Failed to load gallery:', error);
            }
        },

        /**
         * Enhance the current prompt using LLM
         */
        async enhancePrompt() {
            if (!this.imagePrompt.trim()) {
                this.showToast('Please enter a prompt first', 'error');
                return;
            }

            this.isEnhancing = true;

            try {
                const response = await fetch('/api/enhance', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: this.imagePrompt })
                });

                if (response.ok) {
                    const data = await response.json();
                    this.imagePrompt = data.enhanced;
                    this.showToast('Prompt enhanced!');
                } else {
                    const error = await response.json();
                    throw new Error(error.detail || 'Enhancement failed');
                }
            } catch (error) {
                console.error('Enhancement error:', error);
                this.showToast(error.message, 'error');
            } finally {
                this.isEnhancing = false;
            }
        },

        /**
         * Enhance the video prompt using LLM (WAN 2.2 optimized)
         */
        async enhanceVideoPrompt() {
            if (!this.videoPrompt.trim()) {
                this.showToast('Please enter a prompt first', 'error');
                return;
            }

            this.isEnhancingVideo = true;

            try {
                const response = await fetch('/api/enhance-video', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: this.videoPrompt })
                });

                if (response.ok) {
                    const data = await response.json();
                    this.videoPrompt = data.enhanced;
                    this.showToast('Video prompt enhanced!');
                } else {
                    const error = await response.json();
                    throw new Error(error.detail || 'Enhancement failed');
                }
            } catch (error) {
                console.error('Video enhancement error:', error);
                this.showToast(error.message, 'error');
            } finally {
                this.isEnhancingVideo = false;
            }
        },

        /**
         * Generate an image
         */
        async generateImage() {
            if (!this.imagePrompt.trim()) {
                this.showToast('Please enter a prompt', 'error');
                return;
            }

            this.isGenerating = true;
            this.progress = 0;
            this.progressText = 'Starting generation...';
            this.statusMessage = '';
            this.currentImage = null;

            try {
                // Submit generation request
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: this.imagePrompt,
                        seed: this.imageSettings.seed,
                        steps: this.imageSettings.steps,
                        aspect_ratio: this.imageSettings.aspectRatio,
                        num_images: this.imageSettings.numImages
                    })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Generation failed');
                }

                const result = await response.json();
                const { client_id, prompt_id, seed } = result;

                this.progressText = `Seed: ${seed} - Connecting...`;

                // Connect to WebSocket for progress
                await this.connectWebSocket(client_id, prompt_id, 'image');

            } catch (error) {
                console.error('Generation error:', error);
                this.statusMessage = error.message;
                this.statusType = 'error';
                this.showToast(error.message, 'error');
            } finally {
                this.isGenerating = false;
            }
        },

        /**
         * Connect to WebSocket for progress updates
         */
        async connectWebSocket(clientId, promptId, type = 'image') {
            return new Promise((resolve, reject) => {
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${wsProtocol}//${window.location.host}/ws/${clientId}`;

                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    if (type === 'image') {
                        this.progressText = 'Generating...';
                    } else {
                        this.videoProgressText = 'Generating video...';
                    }
                };

                this.ws.onmessage = async (event) => {
                    const data = JSON.parse(event.data);

                    if (data.type === 'progress') {
                        const pct = (data.value / data.max) * 100;
                        if (type === 'image') {
                            this.progress = pct;
                            this.progressText = `Step ${data.value}/${data.max}`;
                        } else {
                            this.videoProgress = pct;
                            this.videoProgressText = `Frame ${data.value}/${data.max}`;
                        }
                    }

                    if (data.type === 'complete') {
                        console.log('Generation complete');
                        this.ws.close();

                        // Fetch the result
                        try {
                            const resultResponse = await fetch(`/api/result/${promptId}?type=${type}`);
                            if (resultResponse.ok) {
                                const resultData = await resultResponse.json();

                                if (type === 'image') {
                                    if (resultData.images && resultData.images.length > 0) {
                                        this.currentImage = 'data:image/png;base64,' + resultData.images[0].data;
                                        this.statusMessage = `Generated ${resultData.count} image(s)`;
                                        this.statusType = 'success';
                                        this.showToast('Image generated!');

                                        // Update gallery
                                        await this.loadGallery();
                                    }
                                } else {
                                    if (resultData.video) {
                                        this.currentVideo = 'data:video/mp4;base64,' + resultData.video;
                                        this.videoStatusMessage = 'Video generated!';
                                        this.videoStatusType = 'success';
                                        this.showToast('Video generated!');
                                    }
                                }
                            }
                        } catch (err) {
                            console.error('Failed to fetch result:', err);
                        }

                        resolve();
                    }

                    if (data.type === 'error') {
                        const errorMsg = data.message || 'Generation failed';
                        if (type === 'image') {
                            this.statusMessage = errorMsg;
                            this.statusType = 'error';
                        } else {
                            this.videoStatusMessage = errorMsg;
                            this.videoStatusType = 'error';
                        }
                        this.showToast(errorMsg, 'error');
                        this.ws.close();
                        reject(new Error(errorMsg));
                    }
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    reject(error);
                };

                this.ws.onclose = () => {
                    console.log('WebSocket closed');
                };
            });
        },

        /**
         * Handle image file selection for I2V
         */
        async handleImageSelect(event) {
            const file = event.target.files[0];
            if (file) {
                await this.uploadImage(file);
            }
        },

        /**
         * Handle image drop for I2V
         */
        async handleImageDrop(event) {
            this.isDragging = false;
            const file = event.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                await this.uploadImage(file);
            }
        },

        /**
         * Upload image to server for I2V
         */
        async uploadImage(file) {
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                this.videoSettings.inputImagePreview = e.target.result;
            };
            reader.readAsDataURL(file);

            // Upload to server
            try {
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/api/upload-image', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    this.videoSettings.inputImage = data.filename;
                    this.showToast('Image uploaded');
                } else {
                    throw new Error('Upload failed');
                }
            } catch (error) {
                console.error('Upload error:', error);
                this.showToast('Failed to upload image', 'error');
            }
        },

        /**
         * Generate a video
         */
        async generateVideo() {
            if (!this.videoPrompt.trim()) {
                this.showToast('Please enter a prompt', 'error');
                return;
            }

            if (!this.videoSettings.modelHigh || !this.videoSettings.modelLow) {
                this.showToast('Please select both high and low noise models', 'error');
                return;
            }

            if (!this.videoSettings.loraHigh || !this.videoSettings.loraLow) {
                this.showToast('Please select both high and low noise LoRAs', 'error');
                return;
            }

            if (!this.videoSettings.inputImage) {
                this.showToast('Please upload an input image', 'error');
                return;
            }

            this.isGeneratingVideo = true;
            this.videoProgress = 0;
            this.videoProgressText = 'Starting generation...';
            this.videoStatusMessage = '';
            this.currentVideo = null;

            try {
                const response = await fetch('/api/generate-video', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        mode: 'i2v',
                        prompt: this.videoPrompt,
                        seed: this.videoSettings.seed,
                        resolution: this.videoSettings.resolution,
                        frames: this.videoSettings.frames,
                        model_high: this.videoSettings.modelHigh,
                        model_low: this.videoSettings.modelLow,
                        lora_high: this.videoSettings.loraHigh,
                        lora_low: this.videoSettings.loraLow,
                        input_image: this.videoSettings.inputImage,
                        upscale_enabled: this.videoSettings.upscaleEnabled,
                        upscale_model: this.videoSettings.upscaleModel,
                        interpolate_enabled: this.videoSettings.interpolateEnabled,
                        interpolate_model: this.videoSettings.interpolateModel,
                        interpolate_multiplier: this.videoSettings.interpolateMultiplier
                    })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Generation failed');
                }

                const result = await response.json();
                const { client_id, prompt_id, seed } = result;

                this.videoProgressText = `Seed: ${seed} - Connecting...`;

                // Connect to WebSocket for progress
                await this.connectWebSocket(client_id, prompt_id, 'video');

            } catch (error) {
                console.error('Video generation error:', error);
                this.videoStatusMessage = error.message;
                this.videoStatusType = 'error';
                this.showToast(error.message, 'error');
            } finally {
                this.isGeneratingVideo = false;
            }
        }
    };
}

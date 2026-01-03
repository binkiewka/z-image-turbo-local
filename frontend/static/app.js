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

        // Fullscreen state
        showFullscreen: false,
        fullscreenImage: '',

        // Image settings
        imageSettings: {
            aspectRatio: '1:1',
            seed: -1,
            steps: 8,
            cfg: 1.0,
            sampler: 'euler',
            scheduler: 'sgm_uniform',
            numImages: 1,
            loras: [],
            upscaleEnabled: false,
            upscaleModel: ''
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
            lorasHighUser: [],
            lorasLowUser: [],
            inputImage: null,
            inputImagePreview: null,
            upscaleEnabled: false,
            upscaleModel: '',
            interpolateEnabled: false,
            interpolateModel: '',
            interpolateMultiplier: 2
        },

        videoAccordion: {
            systemModels: true
        },

        // System state
        isConnected: false,
        statusInterval: null,
        showVideoModels: true,
        showEnhancement: false,
        isDragging: false,

        models: {
            video: { high: [], low: [], loras_high: [], loras_low: [], style_loras: [] },
            enhancement: { upscale: [], vfi: [] },
            image: { aspect_ratios: [], loras: [] },
            samplers: [
                "euler", "euler_ancestral", "heun", "heunpp2",
                "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
                "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
                "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm"
            ],
            schedulers: [
                "normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"
            ]
        },

        // Toast notifications
        toasts: [],
        toastId: 0,

        // WebSocket connection
        ws: null,

        // Character mode state
        characterMode: false,
        characterSession: null,
        characterConcept: '',
        characterTemplatePreset: 'photorealistic',
        isCreatingCharacter: false,
        isGeneratingSheet: false,
        characterSessions: [],
        newTurnPrompt: '',
        characterTemplatePresets: [
            'photorealistic', 'character_design', 'comic_american', 
            'anime_ghibli', 'portrait_studio', 'fantasy_art'
        ],
        
        characterForm: {
            name: '',
            age: '',
            gender: 'Female',
            ethnicity: '',
            build: 'Average',
            faceShape: 'Oval',
            skin: '',
            eyeColor: '',
            eyeShape: '',
            eyebrows: '',
            nose: '',
            lips: '',
            expression: 'Neutral',
            hairColor: '',
            hairLength: 'Medium',
            hairStyle: '',
            hairTexture: 'Straight',
            distinguishingFeatures: '',
            defaultAttire: ''
        },
        
        characterOptions: {
            genders: ['Female', 'Male', 'Non-binary', 'Other'],
            builds: ['Slim', 'Athletic', 'Average', 'Curvy', 'Muscular', 'Heavyset'],
            faceShapes: ['Oval', 'Round', 'Square', 'Heart', 'Oblong', 'Diamond'],
            expressions: ['Neutral', 'Smiling', 'Serious', 'Contemplative', 'Confident', 'Mysterious'],
            hairLengths: ['Bald/Shaved', 'Buzz cut', 'Short', 'Medium', 'Long', 'Very long'],
            hairTextures: ['Straight', 'Wavy', 'Curly', 'Coily', 'Braided', 'Dreadlocks']
        },

        /**
         * Initialize the application
         */
        async init() {
            // Calculate initial frames from duration
            this.videoSettings.frames = this.videoDuration * 16 + 1;

            this.$watch('videoDuration', (value) => {
                this.videoSettings.frames = value * 16 + 1;
            });

            this.$watch('videoSettings.lorasHighUser.length', (newLen, oldLen) => {
                if (newLen > 0 && oldLen === 0) {
                    this.videoAccordion.systemModels = false;
                }
            });

            this.$watch('videoSettings.lorasLowUser.length', (newLen, oldLen) => {
                if (newLen > 0 && oldLen === 0) {
                    this.videoAccordion.systemModels = false;
                }
            });

            // Start status polling
            this.checkStatus();
            this.statusInterval = setInterval(() => this.checkStatus(), 5000);

            // Load models
            await this.refreshModels();

            // Load gallery from server
            await this.loadGallery();
        },

        /**
         * Check backend connection status
         */
        async checkStatus() {
            try {
                const response = await fetch('/api/health');
                this.isConnected = response.ok;
            } catch (error) {
                this.isConnected = false;
            }
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

                    this.models = {
                        video: {
                            high: data.video?.high || [],
                            low: data.video?.low || [],
                            loras_high: data.video?.loras_high || [],
                            loras_low: data.video?.loras_low || [],
                            style_loras: data.video?.style_loras || []
                        },
                        enhancement: {
                            upscale: data.enhancement?.upscale || [],
                            vfi: data.enhancement?.vfi || []
                        },
                        image: {
                            aspect_ratios: data.image?.aspect_ratios || [],
                            loras: data.image?.loras || []
                        },
                        samplers: data.samplers || [],
                        schedulers: data.schedulers || []
                    };

                    // Default Upscale Model for Image if available
                    if (this.models.enhancement.upscale.length > 0 && !this.imageSettings.upscaleModel) {
                        this.imageSettings.upscaleModel = this.models.enhancement.upscale[0];
                    }



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
                // Silent fail
            }
        },

        /**
         * Delete an image
         */
        async deleteImage(filename) {
            if (!confirm('Are you sure you want to delete this image?')) return;

            // Get just the filename if full path provided
            const name = filename.split('/').pop();

            try {
                const response = await fetch(`/api/gallery/${name}`, {
                    method: 'DELETE'
                });

                if (response.ok) {
                    this.showToast('Image deleted');
                    await this.loadGallery();

                    // Clear current image if it was the deleted one
                    if (this.currentImage && this.currentImage.includes(name)) {
                        this.currentImage = null;
                    }
                } else {
                    throw new Error('Deletion failed');
                }
            } catch (error) {
                this.showToast('Failed to delete image', 'error');
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
                        seed: this.imageSettings.seed,
                        steps: this.imageSettings.steps,
                        cfg: parseFloat(this.imageSettings.cfg),
                        sampler_name: this.imageSettings.sampler,
                        scheduler: this.imageSettings.scheduler,
                        aspect_ratio: this.imageSettings.aspectRatio,
                        num_images: this.imageSettings.numImages,
                        loras: this.imageSettings.loras.filter(l => l.name.length > 0).map(l => ({
                            name: l.name,
                            strength_model: parseFloat(l.strength_model),
                            strength_clip: parseFloat(l.strength_clip)
                        })),
                        upscale_enabled: this.imageSettings.upscaleEnabled,
                        upscale_model: this.imageSettings.upscaleEnabled ? this.imageSettings.upscaleModel : null
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
                            // Ignore progress events for video to prevent "Frame X/Y" overwriting status
                            // We rely purely on 'status' messages for video
                        }
                    }

                    if (data.type === 'status') {
                        if (type === 'video') {
                            this.videoProgressText = data.message;
                        } else {
                            this.progressText = data.message;
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
                                        this.currentImage = resultData.images[0].url;
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
                        loras_high_user: this.videoSettings.lorasHighUser.filter(l => l.name).map(l => ({
                            name: l.name,
                            strength_model: parseFloat(l.strength_model)
                        })),
                        loras_low_user: this.videoSettings.lorasLowUser.filter(l => l.name).map(l => ({
                            name: l.name,
                            strength_model: parseFloat(l.strength_model)
                        })),
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
        },

        addLora() {
            this.imageSettings.loras.push({ name: '', strength_model: 1.0, strength_clip: 1.0 });
        },

        removeLora(index) {
            this.imageSettings.loras.splice(index, 1);
        },

        addVideoLoraHigh() {
            this.videoSettings.lorasHighUser.push({ name: '', strength_model: 1.0 });
        },

        removeVideoLoraHigh(index) {
            this.videoSettings.lorasHighUser.splice(index, 1);
        },

        addVideoLoraLow() {
            this.videoSettings.lorasLowUser.push({ name: '', strength_model: 1.0 });
        },

        removeVideoLoraLow(index) {
            this.videoSettings.lorasLowUser.splice(index, 1);
        },

        /**
         * Open image in fullscreen
         */
        openFullscreen(url) {
            this.fullscreenImage = url;
            this.showFullscreen = true;
        },

        toggleCharacterMode() {
            this.characterMode = !this.characterMode;
            if (this.characterMode) {
                this.loadCharacterSessions();
            }
        },

        async loadCharacterSessions() {
            try {
                const response = await fetch('/api/character');
                if (response.ok) {
                    const data = await response.json();
                    this.characterSessions = data.sessions || [];
                }
            } catch (error) {
                console.error('Failed to load character sessions:', error);
            }
        },

        buildCharacterSheet() {
            const f = this.characterForm;
            let sheet = `## Core Identity\n`;
            sheet += `- Name: ${f.name || 'Unknown'}\n`;
            sheet += `- Age: ${f.age || 'Adult'}\n`;
            sheet += `- Gender: ${f.gender || 'Unspecified'}\n`;
            sheet += `- Ethnicity: ${f.ethnicity || 'Unspecified'}\n`;
            sheet += `- Build: ${f.build || 'Average'}\n\n`;
            
            sheet += `## Face & Features\n`;
            sheet += `- Face Shape: ${f.faceShape || 'Oval'}\n`;
            sheet += `- Skin: ${f.skin || 'Natural skin tone'}\n`;
            sheet += `- Eyes: ${f.eyeColor}${f.eyeShape ? ', ' + f.eyeShape : ''}\n`;
            sheet += `- Eyebrows: ${f.eyebrows || 'Natural'}\n`;
            sheet += `- Nose: ${f.nose || 'Proportionate'}\n`;
            sheet += `- Lips: ${f.lips || 'Natural'}\n`;
            sheet += `- Expression: ${f.expression || 'Neutral'}\n\n`;
            
            sheet += `## Hair\n`;
            sheet += `- Color: ${f.hairColor || 'Natural'}\n`;
            sheet += `- Length: ${f.hairLength || 'Medium'}\n`;
            sheet += `- Style: ${f.hairStyle || 'Natural'}\n`;
            sheet += `- Texture: ${f.hairTexture || 'Straight'}\n\n`;
            
            sheet += `## Distinguishing Features\n`;
            sheet += f.distinguishingFeatures ? `${f.distinguishingFeatures}\n\n` : `- None specified\n\n`;
            
            sheet += `## Default Attire\n`;
            sheet += f.defaultAttire ? `${f.defaultAttire}\n` : `- Casual clothing\n`;
            
            return sheet;
        },

        async generateCharacterFromConcept() {
            if (!this.characterConcept.trim()) {
                this.showToast('Please enter a character concept', 'error');
                return;
            }

            this.isGeneratingSheet = true;
            try {
                const response = await fetch('/api/character/generate-structured', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ concept: this.characterConcept })
                });

                if (response.ok) {
                    const data = await response.json();
                    if (data.character) {
                        const c = data.character;
                        this.characterForm.name = c.name || '';
                        this.characterForm.age = c.age || '';
                        this.characterForm.gender = c.gender || 'Female';
                        this.characterForm.ethnicity = c.ethnicity || '';
                        this.characterForm.build = c.build || 'Average';
                        this.characterForm.faceShape = c.face_shape || 'Oval';
                        this.characterForm.skin = c.skin || '';
                        this.characterForm.eyeColor = c.eye_color || '';
                        this.characterForm.eyeShape = c.eye_shape || '';
                        this.characterForm.eyebrows = c.eyebrows || '';
                        this.characterForm.nose = c.nose || '';
                        this.characterForm.lips = c.lips || '';
                        this.characterForm.expression = c.expression || 'Neutral';
                        this.characterForm.hairColor = c.hair_color || '';
                        this.characterForm.hairLength = c.hair_length || 'Medium';
                        this.characterForm.hairStyle = c.hair_style || '';
                        this.characterForm.hairTexture = c.hair_texture || 'Straight';
                        this.characterForm.distinguishingFeatures = c.distinguishing_features || '';
                        this.characterForm.defaultAttire = c.default_attire || '';
                        this.showToast('Character generated! Review and edit as needed.');
                    }
                } else {
                    throw new Error('Generation failed');
                }
            } catch (error) {
                this.showToast(error.message, 'error');
            } finally {
                this.isGeneratingSheet = false;
            }
        },

        async generateCharacterSheet() {
            await this.generateCharacterFromConcept();
        },

        async createCharacter() {
            if (!this.characterForm.name.trim()) {
                this.showToast('Please enter a character name', 'error');
                return;
            }

            this.isCreatingCharacter = true;
            try {
                const description = this.buildCharacterSheet();
                const response = await fetch('/api/character/create', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: this.characterForm.name,
                        description: description,
                        template_preset: this.characterTemplatePreset,
                        aspect_ratio: this.imageSettings.aspectRatio
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    this.characterSession = data.session;
                    this.showToast(`Character "${this.characterForm.name}" created!`);
                    await this.loadCharacterSessions();
                } else {
                    throw new Error('Creation failed');
                }
            } catch (error) {
                this.showToast(error.message, 'error');
            } finally {
                this.isCreatingCharacter = false;
            }
        },

        async loadCharacterSession(sessionId) {
            try {
                const response = await fetch(`/api/character/${sessionId}`);
                if (response.ok) {
                    const data = await response.json();
                    this.characterSession = data.session;
                    this.characterName = data.session.name;
                    this.showToast(`Loaded "${data.session.name}"`);
                }
            } catch (error) {
                this.showToast('Failed to load session', 'error');
            }
        },

        async addCharacterTurn() {
            if (!this.newTurnPrompt.trim() || !this.characterSession) {
                this.showToast('Please enter a modification', 'error');
                return;
            }

            try {
                const response = await fetch(`/api/character/${this.characterSession.id}/turn`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_prompt: this.newTurnPrompt,
                        auto_think: true
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    this.characterSession = data.session;
                    this.newTurnPrompt = '';
                    this.showToast('Modification added');
                }
            } catch (error) {
                this.showToast('Failed to add turn', 'error');
            }
        },

        async generateCharacterImage() {
            if (!this.characterSession) {
                this.showToast('No character session active', 'error');
                return;
            }

            this.isGenerating = true;
            this.progress = 0;
            this.progressText = 'Starting character generation...';
            this.currentImage = null;

            try {
                const response = await fetch(`/api/character/${this.characterSession.id}/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        seed: this.imageSettings.seed,
                        steps: this.imageSettings.steps,
                        cfg: parseFloat(this.imageSettings.cfg),
                        sampler_name: this.imageSettings.sampler,
                        scheduler: this.imageSettings.scheduler,
                        upscale_enabled: this.imageSettings.upscaleEnabled,
                        upscale_model: this.imageSettings.upscaleEnabled ? this.imageSettings.upscaleModel : null,
                        denoise: 0.65  // Lower denoise for img2img to preserve pose/composition
                    })
                });

                if (!response.ok) {
                    throw new Error('Generation failed');
                }

                const result = await response.json();
                const { client_id, prompt_id, seed, is_img2img, denoise } = result;

                const modeText = is_img2img ? `img2img (denoise: ${denoise})` : 'txt2img';
                this.progressText = `Seed: ${seed} - ${modeText} - Connecting...`;
                
                // Connect to WebSocket and handle character-specific completion
                await this.connectCharacterWebSocket(client_id, prompt_id, this.characterSession.id);

            } catch (error) {
                this.showToast(error.message, 'error');
            } finally {
                this.isGenerating = false;
            }
        },

        /**
         * WebSocket handler for character generation (saves last image for img2img)
         */
        async connectCharacterWebSocket(clientId, promptId, sessionId) {
            return new Promise((resolve, reject) => {
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${wsProtocol}//${window.location.host}/ws/${clientId}`;

                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => {
                    console.log('WebSocket connected (character)');
                    this.progressText = 'Generating character...';
                };

                this.ws.onmessage = async (event) => {
                    const data = JSON.parse(event.data);

                    if (data.type === 'progress') {
                        const pct = (data.value / data.max) * 100;
                        this.progress = pct;
                        this.progressText = `Step ${data.value}/${data.max}`;
                    }

                    if (data.type === 'status') {
                        this.progressText = data.message;
                    }

                    if (data.type === 'complete') {
                        console.log('Character generation complete');
                        this.ws.close();

                        try {
                            const resultResponse = await fetch(`/api/result/${promptId}?type=image`);
                            if (resultResponse.ok) {
                                const resultData = await resultResponse.json();

                                if (resultData.images && resultData.images.length > 0) {
                                    this.currentImage = resultData.images[0].url;
                                    this.statusMessage = 'Character generated!';
                                    this.statusType = 'success';
                                    this.showToast('Character image generated!');

                                    // Extract filename from URL and save to session for img2img
                                    const imageUrl = resultData.images[0].url;
                                    const filename = imageUrl.split('/').pop();
                                    
                                    // Save the image filename for next img2img iteration
                                    await fetch(`/api/character/${sessionId}/set-image`, {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ image_filename: filename })
                                    });

                                    await this.loadGallery();
                                }
                            }
                        } catch (err) {
                            console.error('Failed to fetch character result:', err);
                        }

                        resolve();
                    }

                    if (data.type === 'error') {
                        const errorMsg = data.message || 'Generation failed';
                        this.statusMessage = errorMsg;
                        this.statusType = 'error';
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

        async deleteCharacterSession(sessionId) {
            if (!confirm('Delete this character session?')) return;

            try {
                const response = await fetch(`/api/character/${sessionId}`, {
                    method: 'DELETE'
                });

                if (response.ok) {
                    this.showToast('Session deleted');
                    if (this.characterSession?.id === sessionId) {
                        this.characterSession = null;
                    }
                    await this.loadCharacterSessions();
                }
            } catch (error) {
                this.showToast('Failed to delete session', 'error');
            }
        },

        clearCharacterSession() {
            this.characterSession = null;
            this.characterConcept = '';
            this.newTurnPrompt = '';
            this.characterForm = {
                name: '',
                age: '',
                gender: 'Female',
                ethnicity: '',
                build: 'Average',
                faceShape: 'Oval',
                skin: '',
                eyeColor: '',
                eyeShape: '',
                eyebrows: '',
                nose: '',
                lips: '',
                expression: 'Neutral',
                hairColor: '',
                hairLength: 'Medium',
                hairStyle: '',
                hairTexture: 'Straight',
                distinguishingFeatures: '',
                defaultAttire: ''
            };
        }
    };
}

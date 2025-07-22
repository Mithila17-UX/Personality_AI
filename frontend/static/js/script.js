// DOM Elements
const form = document.getElementById('personality-form');
const resultsSection = document.getElementById('results-section');
const loadingOverlay = document.getElementById('loading-overlay');

// Slider elements
const sliders = {
    'time_spent_alone': document.getElementById('time_spent_alone'),
    'social_event_attendance': document.getElementById('social_event_attendance'),
    'going_outside': document.getElementById('going_outside'),
    'friends_circle_size': document.getElementById('friends_circle_size'),
    'post_frequency': document.getElementById('post_frequency')
};

// Initialize sliders
Object.keys(sliders).forEach(key => {
    const slider = sliders[key];
    const valueElement = document.getElementById(`${key}_value`);
    
    if (slider && valueElement) {
        // Set initial value
        updateSliderValue(slider, valueElement);
        
        // Add event listener
        slider.addEventListener('input', () => {
            updateSliderValue(slider, valueElement);
        });
    }
});

function updateSliderValue(slider, valueElement) {
    const value = slider.value;
    if (slider.id === 'time_spent_alone') {
        valueElement.textContent = `${value}h`;
    } else {
        valueElement.textContent = value;
    }
}

// Form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    
    // Collect form data
    const formData = new FormData(form);
    const data = {
        time_spent_alone: formData.get('time_spent_alone'),
        stage_fear: formData.get('stage_fear'),
        social_event_attendance: formData.get('social_event_attendance'),
        going_outside: formData.get('going_outside'),
        drained_after_socializing: formData.get('drained_after_socializing'),
        friends_circle_size: formData.get('friends_circle_size'),
        post_frequency: formData.get('post_frequency')
    };
    
    try {
        // Send prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Hide loading overlay
            loadingOverlay.style.display = 'none';
            
            // Display results
            displayResults(result);
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        } else {
            throw new Error(result.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        loadingOverlay.style.display = 'none';
        alert('An error occurred while processing your request. Please try again.');
    }
});

function displayResults(result) {
    // Update confidence
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceText = document.getElementById('confidence-text');
    const confidencePercentage = Math.round(result.confidence * 100);
    
    confidenceFill.style.width = `${confidencePercentage}%`;
    confidenceText.textContent = `${confidencePercentage}%`;
    
    // Update probabilities - ensure correct mapping
    const introvertPercentage = Math.round(result.introvert_probability * 100);
    const extrovertPercentage = Math.round(result.extrovert_probability * 100);
    
    // Update introvert card
    document.getElementById('introvert-probability').style.width = `${introvertPercentage}%`;
    document.getElementById('introvert-percentage').textContent = `${introvertPercentage}%`;
    
    // Update extrovert card
    document.getElementById('extrovert-probability').style.width = `${extrovertPercentage}%`;
    document.getElementById('extrovert-percentage').textContent = `${extrovertPercentage}%`;
    
    // Update prediction badge
    const predictionBadge = document.getElementById('prediction-badge');
    const predictionText = document.getElementById('prediction-text');
    
    predictionText.textContent = result.result;
    
    if (result.result === 'Extrovert') {
        predictionBadge.classList.add('extrovert');
        predictionBadge.querySelector('i').className = 'fas fa-users';
    } else {
        predictionBadge.classList.remove('extrovert');
        predictionBadge.querySelector('i').className = 'fas fa-book';
    }
    
    // Update insights
    const insightsList = document.getElementById('insights-list');
    insightsList.innerHTML = '';
    
    // Add warning if present
    if (result.warning) {
        const warningItem = document.createElement('div');
        warningItem.className = 'insight-item warning';
        warningItem.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${result.warning}`;
        insightsList.appendChild(warningItem);
    }
    
    result.insights.forEach(insight => {
        const insightItem = document.createElement('div');
        insightItem.className = 'insight-item';
        insightItem.textContent = insight;
        insightsList.appendChild(insightItem);
    });
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Add animation classes
    setTimeout(() => {
        const elements = resultsSection.querySelectorAll('.result-card, .prediction-badge, .insight-item');
        elements.forEach((el, index) => {
            setTimeout(() => {
                el.style.opacity = '0';
                el.style.transform = 'translateY(20px)';
                el.style.transition = 'all 0.5s ease';
                
                setTimeout(() => {
                    el.style.opacity = '1';
                    el.style.transform = 'translateY(0)';
                }, 100);
            }, index * 100);
        });
    }, 100);
}

function resetForm() {
    // Reset form values
    form.reset();
    
    // Reset slider values
    Object.keys(sliders).forEach(key => {
        const slider = sliders[key];
        const valueElement = document.getElementById(`${key}_value`);
        if (slider && valueElement) {
            updateSliderValue(slider, valueElement);
        }
    });
    
    // Hide results section
    resultsSection.style.display = 'none';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Add smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add form validation
function validateForm() {
    const requiredFields = form.querySelectorAll('input[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value) {
            field.classList.add('error');
            isValid = false;
        } else {
            field.classList.remove('error');
        }
    });
    
    return isValid;
}

// Add visual feedback for form interactions
form.querySelectorAll('input, select').forEach(input => {
    input.addEventListener('focus', () => {
        input.parentElement.classList.add('focused');
    });
    
    input.addEventListener('blur', () => {
        input.parentElement.classList.remove('focused');
    });
});

// Add keyboard navigation for sliders
Object.keys(sliders).forEach(key => {
    const slider = sliders[key];
    if (slider) {
        slider.addEventListener('keydown', (e) => {
            const step = parseInt(slider.step) || 1;
            const min = parseInt(slider.min);
            const max = parseInt(slider.max);
            let newValue = parseInt(slider.value);
            
            switch(e.key) {
                case 'ArrowLeft':
                case 'ArrowDown':
                    newValue = Math.max(min, newValue - step);
                    break;
                case 'ArrowRight':
                case 'ArrowUp':
                    newValue = Math.min(max, newValue + step);
                    break;
                default:
                    return;
            }
            
            slider.value = newValue;
            updateSliderValue(slider, document.getElementById(`${key}_value`));
            e.preventDefault();
        });
    }
});

// Add accessibility improvements
document.addEventListener('DOMContentLoaded', () => {
    // Add ARIA labels
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        const label = slider.previousElementSibling;
        if (label) {
            slider.setAttribute('aria-labelledby', label.id || 'slider-label');
        }
    });
    
    // Add focus indicators
    const focusableElements = document.querySelectorAll('button, input, select, textarea, a[href]');
    focusableElements.forEach(element => {
        element.addEventListener('focus', () => {
            element.style.outline = '2px solid #667eea';
            element.style.outlineOffset = '2px';
        });
        
        element.addEventListener('blur', () => {
            element.style.outline = '';
            element.style.outlineOffset = '';
        });
    });
});

// Add loading state management
function showLoading() {
    loadingOverlay.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
    document.body.style.overflow = '';
}

// Add error handling with user-friendly messages
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #ff6b6b;
        color: white;
        padding: 1rem 2rem;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 3000;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .form-group.focused {
        transform: scale(1.02);
        transition: transform 0.2s ease;
    }
    
    .error {
        border-color: #ff6b6b !important;
        box-shadow: 0 0 0 2px rgba(255, 107, 107, 0.2) !important;
    }
`;
document.head.appendChild(style); 
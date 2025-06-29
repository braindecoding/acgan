import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from scipy import signal
from scipy.signal import butter, filtfilt
import os
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class MindBigDataProcessor:
    """
    Processor untuk dataset MindBigData dengan format:
    [id][event][device][channel][code][size][data]
    """
    
    def __init__(self):
        self.device_channels = {
            'MW': ['FP1'],
            'EP': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
            'MU': ['TP9', 'FP1', 'FP2', 'TP10'],
            'IN': ['AF3', 'AF4', 'T7', 'T8', 'PZ']
        }
        self.target_channels = ['T7', 'P7', 'T8', 'P8']  # Channels untuk visual processing
        self.sampling_rates = {'MW': 512, 'EP': 128, 'MU': 220, 'IN': 128}
        self.scaler = StandardScaler()
        
    def parse_line(self, line):
        """Parse satu baris data MindBigData"""
        try:
            parts = line.strip().split('\t')
            if len(parts) != 7:
                return None
                
            id_val = int(parts[0])
            event = int(parts[1])
            device = parts[2]
            channel = parts[3]
            code = int(parts[4])
            size = int(parts[5])
            
            # Parse data values
            data_str = parts[6]
            if device in ['MW', 'MU']:
                data = [int(x) for x in data_str.split(',')]
            else:  # EP, IN
                data = [float(x) for x in data_str.split(',')]
                
            return {
                'id': id_val,
                'event': event,
                'device': device,
                'channel': channel,
                'code': code,
                'size': size,
                'data': np.array(data)
            }
        except Exception as e:
            print(f"Error parsing line: {e}")
            return None
    
    def load_data(self, file_path):
        """Load data dari file MindBigData"""
        print(f"Loading data from {file_path}...")
        
        data_list = []
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num % 10000 == 0:
                    print(f"Processed {line_num} lines...")
                    
                parsed = self.parse_line(line)
                if parsed and parsed['code'] >= 0:  # Skip random signals (-1)
                    data_list.append(parsed)
        
        print(f"Loaded {len(data_list)} valid signals")
        return data_list
    
    def preprocess_signal(self, signal_data, fs=128):
        """Preprocess EEG signal dengan filtering dan normalization"""
        
        # Notch filter untuk 50Hz line noise
        f_notch = 50
        Q = 30
        b_notch, a_notch = signal.iirnotch(f_notch, Q, fs)
        filtered_signal = signal.filtfilt(b_notch, a_notch, signal_data)
        
        # Bandpass filter 0.4-60 Hz
        low_freq = 0.4
        high_freq = 60
        nyquist = fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        if high >= 1.0:
            high = 0.99
            
        b, a = butter(5, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, filtered_signal)
        
        # Remove artifacts (threshold based)
        threshold = 100  # microvolts
        if np.max(np.abs(filtered_signal)) > threshold:
            return None
            
        return filtered_signal
    
    def apply_car(self, signals_dict):
        """Apply Common Average Reference"""
        channels = list(signals_dict.keys())
        signals_array = np.array([signals_dict[ch] for ch in channels])
        
        # Calculate common average
        car_signal = np.mean(signals_array, axis=0)
        
        # Subtract from each channel
        car_applied = {}
        for i, ch in enumerate(channels):
            car_applied[ch] = signals_array[i] - car_signal
            
        return car_applied
    
    def segment_signal(self, signal_data, window_size=32, overlap=4):
        """Segment signal menggunakan sliding window"""
        segments = []
        step = window_size - overlap
        
        for i in range(0, len(signal_data) - window_size + 1, step):
            segment = signal_data[i:i + window_size]
            segments.append(segment)
            
        return np.array(segments)
    
    def prepare_dataset(self, data_list, target_fs=128):
        """Prepare dataset untuk training"""
        print("Preparing dataset...")
        
        # Group data by event
        events = defaultdict(list)
        for item in data_list:
            if item['channel'] in self.target_channels:
                events[item['event']].append(item)
        
        X_data = []
        y_data = []
        
        for event_id, event_data in events.items():
            if len(event_data) < len(self.target_channels):
                continue
                
            # Process each channel
            channels_data = {}
            code = None
            
            for item in event_data:
                if item['channel'] in self.target_channels:
                    # Resample if needed
                    original_fs = self.sampling_rates.get(item['device'], 128)
                    if original_fs != target_fs:
                        num_samples = int(len(item['data']) * target_fs / original_fs)
                        resampled = signal.resample(item['data'], num_samples)
                    else:
                        resampled = item['data']
                    
                    # Preprocess
                    processed = self.preprocess_signal(resampled, target_fs)
                    if processed is not None:
                        channels_data[item['channel']] = processed
                        code = item['code']
            
            # Check if we have all target channels
            if len(channels_data) == len(self.target_channels) and code is not None:
                # Apply CAR
                car_applied = self.apply_car(channels_data)
                
                # Create multi-channel signal matrix
                signal_matrix = []
                for ch in self.target_channels:
                    if ch in car_applied:
                        signal_matrix.append(car_applied[ch])
                
                if len(signal_matrix) == len(self.target_channels):
                    signal_matrix = np.array(signal_matrix)
                    
                    # Segment signals
                    segments_per_channel = []
                    min_segments = float('inf')
                    
                    for i in range(len(self.target_channels)):
                        segments = self.segment_signal(signal_matrix[i])
                        segments_per_channel.append(segments)
                        min_segments = min(min_segments, len(segments))
                    
                    # Take same number of segments from each channel
                    for seg_idx in range(min_segments):
                        multi_channel_segment = []
                        for ch_idx in range(len(self.target_channels)):
                            multi_channel_segment.append(segments_per_channel[ch_idx][seg_idx])
                        
                        X_data.append(np.array(multi_channel_segment))
                        y_data.append(code)
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"Dataset prepared: {X_data.shape}, Labels: {len(np.unique(y_data))}")
        return X_data, y_data


class EEGClassifier:
    """CNN Classifier untuk EEG signals"""
    
    def __init__(self, input_shape, num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.encoder_model = None
        
    def build_model(self):
        """Build CNN model untuk EEG classification"""

        inputs = layers.Input(shape=self.input_shape)

        # Reshape untuk 2D convolution (channels, time, 1)
        x = layers.Reshape((*self.input_shape, 1))(inputs)

        # Batch normalization
        x = layers.BatchNormalization()(x)

        # Temporal convolution - disesuaikan dengan input shape
        x = layers.Conv2D(64, (1, 8), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Spatial convolution - disesuaikan dengan jumlah channel
        x = layers.Conv2D(64, (4, 1), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((1, 2))(x)

        # Additional conv layers dengan kernel size yang sesuai
        x = layers.Conv2D(128, (1, 4), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((1, 2))(x)

        x = layers.Conv2D(128, (1, 4), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Global Average Pooling untuk mengurangi parameter
        x = layers.GlobalAveragePooling2D()(x)

        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)

        # Latent representation
        latent = layers.Dense(128, activation='relu', name='latent')(x)
        latent = layers.Dropout(0.2)(latent)
        latent = layers.BatchNormalization(name='latent_bn')(latent)

        # Classification output
        outputs = layers.Dense(self.num_classes, activation='softmax', name='classification')(latent)

        self.model = keras.Model(inputs, outputs)
        self.encoder_model = keras.Model(inputs, latent)

        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """Train the model"""

        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_test, predicted_classes))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return predicted_classes
    
    def extract_features(self, X):
        """Extract latent features"""
        return self.encoder_model.predict(X)


class ACGAN:
    """Auxiliary Classifier GAN untuk image reconstruction"""
    
    def __init__(self, latent_dim=128, num_classes=10, img_shape=(28, 28, 1)):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_shape = img_shape
        
        self.generator = None
        self.discriminator = None
        self.combined = None
        
    def build_generator(self, use_modulation=False):
        """Build generator network"""
        
        # Class label input
        label = layers.Input(shape=(1,), name='class_label')
        label_embedding = layers.Embedding(self.num_classes, 128)(label)
        label_embedding = layers.Flatten()(label_embedding)
        
        # EEG latent input
        eeg_latent = layers.Input(shape=(self.latent_dim,), name='eeg_latent')
        
        if use_modulation:
            # Modulation layer
            mu = layers.Dense(self.latent_dim, name='mu')(eeg_latent)
            sigma = layers.Dense(self.latent_dim, activation='sigmoid', name='sigma')(eeg_latent)
            modulated = layers.Multiply()([sigma, eeg_latent])
            modulated = layers.Add()([mu, modulated])
            
            # Combine with label
            combined = layers.Multiply()([modulated, label_embedding])
        else:
            # Direct combination
            combined = layers.Multiply()([eeg_latent, label_embedding])
        
        # Generator network
        x = layers.Dense(7 * 7 * 128, activation='relu')(combined)
        x = layers.Reshape((7, 7, 128))(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(1, 3, padding='same', activation='tanh')(x)
        
        self.generator = keras.Model([label, eeg_latent], x)
        return self.generator
    
    def build_discriminator(self):
        """Build discriminator network"""
        
        img = layers.Input(shape=self.img_shape)
        
        x = layers.Conv2D(16, 3, strides=2, padding='same')(img)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(32, 3, strides=2, padding='same')(x)
        x = layers.ZeroPadding2D()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        
        x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        
        x = layers.Conv2D(128, 3, strides=1, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Flatten()(x)
        
        # Validity output
        validity = layers.Dense(1, name='validity')(x)
        
        # Class prediction output
        label = layers.Dense(self.num_classes, activation='softmax', name='label')(x)
        
        self.discriminator = keras.Model(img, [validity, label])
        return self.discriminator
    
    def build_combined(self):
        """Build combined model untuk training generator"""
        
        self.discriminator.trainable = False
        
        # Generator inputs
        label = layers.Input(shape=(1,))
        eeg_latent = layers.Input(shape=(self.latent_dim,))
        
        # Generate image
        fake_img = self.generator([label, eeg_latent])
        
        # Discriminator outputs
        validity, pred_label = self.discriminator(fake_img)
        
        self.combined = keras.Model([label, eeg_latent], [validity, pred_label])
        return self.combined
    
    def compile_models(self):
        """Compile all models"""
        
        # Compile discriminator
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
            metrics=['accuracy']
        )
        
        # Compile combined model
        self.combined.compile(
            optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
            metrics=['accuracy']
        )
    
    def train(self, eeg_features, labels, mnist_images, epochs=500, batch_size=32):
        """Train AC-GAN"""

        # Load MNIST for real images
        (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 127.5 - 1.0
        x_train = np.expand_dims(x_train, axis=3)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        print(f"Starting AC-GAN training for {epochs} epochs...")

        for epoch in range(epochs):

            # Train discriminator
            idx = np.random.randint(0, len(eeg_features), batch_size)
            real_imgs = x_train[idx]
            real_labels = y_train[idx]

            # Generate fake images
            fake_labels = labels[idx]
            fake_features = eeg_features[idx]
            fake_imgs = self.generator.predict([fake_labels, fake_features], verbose=0)

            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(real_imgs, [valid, real_labels])
            d_loss_fake = self.discriminator.train_on_batch(fake_imgs, [fake, fake_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            g_loss = self.combined.train_on_batch([fake_labels, fake_features], [valid, fake_labels])

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, D loss: {d_loss[0]:.4f}, G loss: {g_loss[0]:.4f}")

                # Save sample images
                if epoch % 100 == 0:
                    self.save_sample_images(epoch, eeg_features[:5], labels[:5])
    
    def save_sample_images(self, epoch, features, labels):
        """Save sample generated images"""
        
        generated = self.generator.predict([labels, features])
        generated = 0.5 * generated + 0.5
        
        fig, axes = plt.subplots(1, len(labels), figsize=(15, 3))
        for i in range(len(labels)):
            axes[i].imshow(generated[i, :, :, 0], cmap='gray')
            axes[i].set_title(f'Label: {labels[i]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'generated_epoch_{epoch}.png')
        plt.close()


def calculate_reconstruction_metrics(generated_imgs, target_imgs):
    """Calculate Dice coefficient and SSIM"""
    from skimage.metrics import structural_similarity as ssim
    
    dice_scores = []
    ssim_scores = []
    
    for i in range(len(generated_imgs)):
        gen_img = generated_imgs[i, :, :, 0]
        target_img = target_imgs[i, :, :, 0]
        
        # Binarize for Dice coefficient
        gen_binary = (gen_img > 0.5).astype(int)
        target_binary = (target_img > 0.5).astype(int)
        
        # Dice coefficient
        intersection = np.sum(gen_binary * target_binary)
        dice = (2.0 * intersection) / (np.sum(gen_binary) + np.sum(target_binary) + 1e-8)
        dice_scores.append(dice)
        
        # SSIM
        ssim_score = ssim(gen_img, target_img, data_range=1.0)
        ssim_scores.append(ssim_score)
    
    return np.array(dice_scores), np.array(ssim_scores)


# Main execution pipeline
def main():
    """Main execution function"""
    
    # Initialize processor
    processor = MindBigDataProcessor()
    
    # Load and prepare data
    data_list = processor.load_data('mindbigdata.txt')  # Replace with your file path
    X, y = processor.prepare_dataset(data_list)
    
    if len(X) == 0:
        print("No valid data found!")
        return
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Normalize data
    X_train_norm = processor.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_norm = processor.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_norm = processor.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Build and train classifier
    classifier = EEGClassifier(input_shape=X_train.shape[1:])
    classifier.build_model()
    classifier.compile_model()
    
    print("Training classifier...")
    history = classifier.train(X_train_norm, y_train, X_val_norm, y_val)
    
    # Evaluate classifier
    predictions = classifier.evaluate(X_test_norm, y_test)
    
    # Extract features for GAN
    train_features = classifier.extract_features(X_train_norm)
    test_features = classifier.extract_features(X_test_norm)
    
    # Build and train AC-GAN
    ac_gan = ACGAN()
    ac_gan.build_generator(use_modulation=False)
    ac_gan.build_discriminator()
    ac_gan.build_combined()
    ac_gan.compile_models()
    
    print("Training AC-GAN...")
    ac_gan.train(train_features, y_train, None, epochs=500)
    
    # Generate images and calculate metrics
    generated_images = ac_gan.generator.predict([y_test, test_features])
    
    # Load MNIST for comparison
    (_, _), (x_test_mnist, y_test_mnist) = keras.datasets.mnist.load_data()
    x_test_mnist = x_test_mnist.astype('float32') / 127.5 - 1.0
    x_test_mnist = np.expand_dims(x_test_mnist, axis=3)
    
    # Calculate reconstruction metrics
    dice_scores, ssim_scores = calculate_reconstruction_metrics(generated_images, x_test_mnist[:len(generated_images)])
    
    print(f"\nReconstruction Results:")
    print(f"Average Dice Score: {np.mean(dice_scores):.4f}")
    print(f"Average SSIM Score: {np.mean(ssim_scores):.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    for i in range(10):
        # Original MNIST
        axes[0, i].imshow(x_test_mnist[i, :, :, 0], cmap='gray')
        axes[0, i].set_title(f'Original: {y_test_mnist[i]}')
        axes[0, i].axis('off')
        
        # Generated
        axes[1, i].imshow(generated_images[i, :, :, 0], cmap='gray')
        axes[1, i].set_title(f'Generated: {y_test[i]}')
        axes[1, i].axis('off')
        
        # Difference
        diff = np.abs(x_test_mnist[i, :, :, 0] - generated_images[i, :, :, 0])
        axes[2, i].imshow(diff, cmap='hot')
        axes[2, i].set_title(f'SSIM: {ssim_scores[i]:.3f}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstruction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
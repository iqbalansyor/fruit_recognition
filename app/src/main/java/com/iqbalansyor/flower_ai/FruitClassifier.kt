package com.iqbalansyor.flower_ai

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * FruitClassifier handles loading the TFLite model and running inference.
 *
 * Model specifications (from Netron):
 * - Input: float32[-1, 150, 150, 3] - Image of 150x150 pixels with 3 color channels (RGB)
 * - Output: float32[-1, 6] - 6 fruit classes with probability scores
 */
class FruitClassifier(context: Context) {

    private var interpreter: Interpreter? = null

    // Model input dimensions
    private val inputImageWidth = 150
    private val inputImageHeight = 150
    private val inputChannels = 3 // RGB

    // Number of output classes
    private val numClasses = 6

    // Labels for the 6 fruit/vegetable classes (matching model training order)
    private val labels = listOf(
        "Apple",
        "Banana",
        "Lemon",
        "Onion",
        "Potato",
        "Watermelon"
    )

    init {
        // Load the TFLite model from assets
        interpreter = Interpreter(loadModelFile(context, "model/FruitsClassifier.tflite"))
    }

    /**
     * Loads the TFLite model file from the assets folder.
     * Using memory-mapped file for efficient loading of large models.
     */
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelPath)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Classifies the given bitmap image.
     *
     * @param bitmap The input image to classify
     * @return A Pair of (label, confidence) for the top prediction
     */
    fun classify(bitmap: Bitmap): Pair<String, Float> {
        // Step 1: Resize bitmap to model's expected input size (150x150)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputImageWidth, inputImageHeight, true)

        // Step 2: Convert bitmap to ByteBuffer (model input format)
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

        // Step 3: Prepare output array for 6 classes
        val outputArray = Array(1) { FloatArray(numClasses) }

        // Step 4: Run inference
        interpreter?.run(inputBuffer, outputArray)

        // Step 5: Find the class with highest probability
        val probabilities = outputArray[0]
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val maxProbability = probabilities[maxIndex]

        return Pair(labels[maxIndex], maxProbability)
    }

    /**
     * Converts a Bitmap to a ByteBuffer that the model can process.
     *
     * The conversion process:
     * 1. Allocate a ByteBuffer with size = 1 * 150 * 150 * 3 * 4 bytes (float32)
     * 2. Extract RGB values from each pixel
     * 3. Normalize values to 0-1 range (divide by 255)
     */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        // Calculate buffer size: batch(1) * height * width * channels * bytes_per_float(4)
        val bufferSize = 1 * inputImageHeight * inputImageWidth * inputChannels * 4
        val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        // Extract pixel values
        val pixels = IntArray(inputImageWidth * inputImageHeight)
        bitmap.getPixels(pixels, 0, inputImageWidth, 0, 0, inputImageWidth, inputImageHeight)

        // Convert each pixel to normalized float values
        for (pixel in pixels) {
            // Extract RGB components (Android stores as ARGB)
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF

            // Normalize to 0-1 range and add to buffer
            // Note: If your model was trained with different normalization (e.g., -1 to 1),
            // you'll need to adjust these calculations
            byteBuffer.putFloat(r / 255.0f)
            byteBuffer.putFloat(g / 255.0f)
            byteBuffer.putFloat(b / 255.0f)
        }

        return byteBuffer
    }

    /**
     * Releases the interpreter resources.
     * Call this when the classifier is no longer needed.
     */
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
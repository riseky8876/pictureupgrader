package com.ernesto.pictureupgrader

import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.provider.Settings
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.ernesto.pictureupgrader.databinding.ActivityMainBinding
import com.ernesto.pictureupgrader.databinding.DialogResolutionReductionBinding
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    companion object {
        private const val GALLERY_REQ  = 0x3a
        private const val REQUEST_CODE = 0x3f
        init { System.loadLibrary("pictureupgrader") }
    }

    private lateinit var binding: ActivityMainBinding
    private var dlg: AlertDialog? = null
    private var selectedImagePath: String? = null

    private external fun initCrashHandler(crashLogPath: String)
    private external fun imgSupResolution(inPath: String, outPath: String, modelDir: String): Boolean
    private external fun imgColouration(inPath: String, outPath: String, modelDir: String): Boolean

    private fun modelDir(): File = File(getExternalFilesDir(null), "models")

    // Check that all critical .bin files exist and are non-empty
    private fun modelsReady(): Boolean {
        val dir = modelDir()
        val required = listOf(
            "siggraph17_color_sim.bin",
            "encoder.bin",
            "generator.bin",
            "real_esrgan.bin",
            "scrfd_500m-opt2.bin"
        )
        val missing = required.filter { name ->
            val f = File(dir, name)
            !f.exists() || f.length() < 1024 * 100 // must be at least 100KB
        }
        if (missing.isNotEmpty()) {
            android.util.Log.e("PU", "Missing/small models: $missing")
        }
        return missing.isEmpty()
    }

    private fun showDebugDialog(title: String, msg: String) {
        runOnUiThread {
            val tv = TextView(this).apply {
                text = msg; setPadding(32,16,32,16)
                setTextIsSelectable(true); textSize = 11f
            }
            AlertDialog.Builder(this)
                .setTitle(title)
                .setView(ScrollView(this).apply { addView(tv) })
                .setPositiveButton("OK", null).show()
        }
    }

    // Copy assets/models/* → external storage, always overwrite
    private fun copyModelsFromAssets(): String {
        val dest = modelDir()
        dest.mkdirs()
        val log = StringBuilder()
        val assetList = assets.list("models") ?: return "assets.list returned null"
        log.append("Assets found: ${assetList.size}\n")
        for (name in assetList) {
            if (name == "README.txt" || name.startsWith(".")) continue
            val outFile = File(dest, name)
            try {
                assets.open("models/$name").use { input ->
                    FileOutputStream(outFile).use { output -> input.copyTo(output) }
                }
                log.append("✓ $name (${outFile.length()} bytes)\n")
            } catch (e: Exception) {
                log.append("✗ $name FAILED: ${e.message}\n")
            }
        }
        return log.toString()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Crash handler
        val crashLog = File(getExternalFilesDir(null), "crash.log").absolutePath
        try { initCrashHandler(crashLog) } catch (e: Exception) {}

        // Show previous crash
        val prevCrash = File(getExternalFilesDir(null), "crash.log")
        val prevStep  = File(getExternalFilesDir(null), "crash.log.step")
        if (prevCrash.exists()) {
            val stepTxt  = if (prevStep.exists()) prevStep.readText().trim() else "unknown"
            showDebugDialog("💥 Previous Crash",
                "Last step: $stepTxt\n\nDetail:\n${prevCrash.readText()}")
            prevCrash.delete(); prevStep.delete()
        }

        // Permissions
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.S) {
            if (!Environment.isExternalStorageManager())
                startActivityForResult(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION), REQUEST_CODE)
        } else {
            if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED)
                requestPermissions(arrayOf(android.Manifest.permission.WRITE_EXTERNAL_STORAGE), REQUEST_CODE)
        }

        binding.btnSelect.setOnClickListener {
            startActivityForResult(
                Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI), GALLERY_REQ)
        }

        binding.btnSuperResolution.setOnClickListener {
            selectedImagePath?.let {
                if (!modelsReady()) { copyModelsWithDialog { processImage(::imgSupResolution, "Super Resolution") } }
                else {
                    val bmp = BitmapFactory.decodeFile(it)
                    if ((bmp?.height ?: 0) > 1500 || (bmp?.width ?: 0) > 1500) {
                        AlertDialog.Builder(this)
                            .setTitle(R.string.warn_title).setMessage(R.string.large_warn)
                            .setPositiveButton("Yes") { _, _ -> processImage(::imgSupResolution, "Super Resolution") }
                            .setNegativeButton("No", null).create().show()
                    } else processImage(::imgSupResolution, "Super Resolution")
                }
            } ?: toast("Please select an image first.")
        }

        binding.btnColorization.setOnClickListener {
            selectedImagePath?.let {
                if (!modelsReady()) copyModelsWithDialog { processImage(::imgColouration, "Colouration") }
                else processImage(::imgColouration, "Colouration")
            } ?: toast("Please select an image first.")
        }

        binding.btnDownsmp.setOnClickListener {
            selectedImagePath?.let { imagePath ->
                val drrb = DialogResolutionReductionBinding.inflate(layoutInflater)
                val bitmap = BitmapFactory.decodeFile(imagePath)
                drrb.seekBar.max = 100; drrb.seekBar.progress = 80
                drrb.seekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(s: SeekBar, p: Int, f: Boolean) { drrb.currentValueTextView.text = p.toString() }
                    override fun onStartTrackingTouch(s: SeekBar) {}
                    override fun onStopTrackingTouch(s: SeekBar) {}
                })
                AlertDialog.Builder(this)
                    .setTitle(R.string.down_sampling)
                    .setMessage("${getString(R.string.current_res)}: ${bitmap.width} x ${bitmap.height}")
                    .setView(drrb.root)
                    .setPositiveButton("OK") { _, _ ->
                        thread {
                            try {
                                val scale = drrb.seekBar.progress.toFloat() / 100f
                                val m = android.graphics.Matrix().apply { postScale(scale, scale) }
                                val scaled = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
                                FileOutputStream(File(imagePath)).use { scaled?.compress(Bitmap.CompressFormat.JPEG, 95, it) }
                                runOnUiThread {
                                    binding.imageView.setImageBitmap(BitmapFactory.decodeFile(imagePath))
                                    toast("Down sampling completed!")
                                }
                            } catch (e: IOException) { runOnUiThread { toast("Down sampling failed!") } }
                        }
                    }
                    .setNegativeButton("Cancel", null).create().show()
            } ?: toast("Please select an image first.")
        }

        // Always re-copy models on fresh install or if any bin is missing/small
        if (!modelsReady()) {
            copyModelsWithDialog(onDone = null)
        }
    }

    private fun copyModelsWithDialog(onDone: (() -> Unit)?) {
        val d = AlertDialog.Builder(this)
            .setTitle(R.string.in_progress)
            .setMessage(R.string.extracting)
            .setCancelable(false).create()
        d.show()
        thread {
            val log = copyModelsFromAssets()
            android.util.Log.i("PU", "Copy log:\n$log")
            val ready = modelsReady()
            runOnUiThread {
                d.dismiss()
                if (!ready) {
                    // Show what was copied so user can report
                    val dir = modelDir()
                    val fileList = dir.listFiles()
                        ?.joinToString("\n") { "${it.name} — ${it.length()} bytes" }
                        ?: "empty"
                    showDebugDialog("⚠️ Model Copy Result",
                        "Copy log:\n$log\n\nFiles in ${dir.absolutePath}:\n$fileList")
                } else {
                    onDone?.invoke()
                }
            }
        }
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == GALLERY_REQ && resultCode == RESULT_OK && data != null) {
            val uri = data.data ?: return
            selectedImagePath = contentResolver.query(uri, null, null, null, null)?.use { c ->
                c.moveToNext(); c.getString(c.getColumnIndexOrThrow("_data"))
            }
            binding.imageView.setImageBitmap(BitmapFactory.decodeFile(selectedImagePath))
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE && (grantResults.isEmpty() || grantResults[0] != PackageManager.PERMISSION_GRANTED)) {
            toast("Insufficient permission!"); finish()
        }
    }

    private fun processImagePath(p: String) =
        File(p).let { "${it.parent ?: ""}/${it.nameWithoutExtension}_processed.jpg" }

    private fun processImage(fn: (String, String, String) -> Boolean, name: String) {
        val imagePath = selectedImagePath ?: run { toast("Select an image first."); return }
        dlg = AlertDialog.Builder(this)
            .setTitle(R.string.in_progress)
            .setView(ProgressBar(this))
            .setMessage(getString(R.string.content_in_progress).format(name))
            .setCancelable(false).create()
        dlg?.show()

        thread {
            val outPath   = processImagePath(imagePath)
            val mdir      = modelDir().absolutePath
            val crashFile = File(getExternalFilesDir(null), "crash.log")
            val stepFile  = File(getExternalFilesDir(null), "crash.log.step")
            crashFile.delete(); stepFile.delete()

            // Log model dir contents before running
            val dirContents = modelDir().listFiles()
                ?.joinToString("\n") { "${it.name} ${it.length()}B" } ?: "empty"
            android.util.Log.i("PU", "Model dir:\n$dirContents")

            var success = false
            var ex = ""
            try { success = fn(imagePath, outPath, mdir) }
            catch (e: Throwable) { ex = e.toString() }

            val outFile = File(outPath)
            runOnUiThread {
                dlg?.dismiss()
                if (success && outFile.exists() && outFile.length() > 0) {
                    binding.imageView.setImageBitmap(BitmapFactory.decodeFile(outPath))
                    toast("$name completed.")
                    selectedImagePath = outPath
                } else {
                    val stepTxt = if (stepFile.exists()) stepFile.readText() else "none"
                    val crashTxt = if (crashFile.exists()) crashFile.readText() else "no crash file"
                    showDebugDialog("❌ $name Failed",
                        "Model dir: $mdir\nFiles:\n$dirContents\n\nLast step: $stepTxt\n\nCrash: $crashTxt\n\nException: $ex")
                }
            }
        }
    }
}
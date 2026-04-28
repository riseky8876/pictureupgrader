package com.ernesto.pictureupgrader

import android.content.Context
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
        private const val GALLERY_REQ = 0x3a
        private const val REQUEST_CODE = 0x3f
        init { System.loadLibrary("pictureupgrader") }
    }

    private lateinit var binding: ActivityMainBinding
    private var dlg: AlertDialog? = null
    private var selectedImagePath: String? = null

    private external fun initCrashHandler(crashLogPath: String)
    private external fun imgSupResolution(inPath: String, outPath: String, modelDir: String): Boolean
    private external fun imgColouration(inPath: String, outPath: String, modelDir: String): Boolean

    private fun showDebugDialog(title: String, msg: String) {
        runOnUiThread {
            val tv = TextView(this).apply {
                text = msg; setPadding(32, 16, 32, 16)
                setTextIsSelectable(true); textSize = 11f
            }
            AlertDialog.Builder(this)
                .setTitle(title)
                .setView(ScrollView(this).apply { addView(tv) })
                .setPositiveButton("OK", null)
                .show()
        }
    }

    private fun copyAssetsModels(): File {
        val dest = File(getExternalFilesDir(null), "models")
        if (!dest.exists()) dest.mkdirs()
        val assetList = assets.list("models") ?: return dest
        for (asset in assetList) {
            if (asset.startsWith(".")) continue
            val out = File(dest, asset)
            try {
                assets.open("models/$asset").use { i -> FileOutputStream(out).use { o -> i.copyTo(o) } }
            } catch (e: Exception) {
                android.util.Log.e("PU", "copy failed: $asset ${e.message}")
            }
        }
        return dest
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // install native crash handler
        val crashLog = File(getExternalFilesDir(null), "crash.log").absolutePath
        try { initCrashHandler(crashLog) } catch (e: Exception) {}

        // permissions
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
            selectedImagePath?.let { path ->
                val bmp = BitmapFactory.decodeFile(path)
                if ((bmp?.height ?: 0) > 1500 || (bmp?.width ?: 0) > 1500) {
                    AlertDialog.Builder(this)
                        .setTitle(R.string.warn_title).setMessage(R.string.large_warn)
                        .setCancelable(false)
                        .setPositiveButton("Yes") { _, _ -> processImage(::imgSupResolution, "Super Resolution") }
                        .setNegativeButton("No", null).create().show()
                } else processImage(::imgSupResolution, "Super Resolution")
            } ?: toast("Please select an image first.")
        }

        binding.btnColorization.setOnClickListener {
            selectedImagePath?.let { processImage(::imgColouration, "Colouration") }
                ?: toast("Please select an image first.")
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
                            } catch (e: IOException) {
                                runOnUiThread { toast("Down sampling failed!") }
                            }
                        }
                    }
                    .setNegativeButton("Cancel", null).create().show()
            } ?: toast("Please select an image first.")
        }

        // copy models on first install
        val modelsDir = File(getExternalFilesDir(null), "models")
        if (!modelsDir.exists() || (modelsDir.listFiles()?.none { it.name.endsWith(".bin") } == true)) {
            val d = AlertDialog.Builder(this)
                .setTitle(R.string.in_progress).setMessage(R.string.extracting).setCancelable(false).create()
            d.show()
            thread {
                val dest = copyAssetsModels()
                val required = listOf(
                    "siggraph17_color_sim.param","siggraph17_color_sim.bin",
                    "encoder.param","encoder.bin","generator.param","generator.bin",
                    "real_esrgan.param","real_esrgan.bin",
                    "scrfd_500m-opt2.param","scrfd_500m-opt2.bin"
                )
                val missing = required.filter { !File(dest, it).exists() }
                runOnUiThread {
                    d.dismiss()
                    if (missing.isNotEmpty()) {
                        showDebugDialog("⚠️ Missing Models",
                            "Missing files:\n${missing.joinToString("\n") { "✗ $it" }}\n\n" +
                            "Found:\n${dest.listFiles()?.joinToString("\n") { "${it.name} (${it.length()}B)" } ?: "none"}")
                    }
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

    private fun processImagePath(imagePath: String): String {
        val f = File(imagePath)
        return "${f.parent ?: ""}/${f.nameWithoutExtension}_processed.jpg"
    }

    private fun processImage(fn: (String, String, String) -> Boolean, name: String) {
        val imagePath = selectedImagePath ?: run { toast("Please select an image first."); return }

        val pb = ProgressBar(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        }
        dlg = AlertDialog.Builder(this)
            .setTitle(R.string.in_progress)
            .setView(pb)
            .setMessage(getString(R.string.content_in_progress).format(name))
            .setCancelable(false).create()
        dlg?.show()

        thread {
            val outPath   = processImagePath(imagePath)
            val modelDir  = File(getExternalFilesDir(null), "models").absolutePath
            val crashFile = File(getExternalFilesDir(null), "crash.log")
            crashFile.delete() // clear previous crash

            // build a pre-run diagnostic string
            val diag = StringBuilder()
            diag.append("Feature: $name\n")
            diag.append("Input:   $imagePath (exists=${File(imagePath).exists()}, ${File(imagePath).length()}B)\n")
            diag.append("Output:  $outPath\n")
            diag.append("Models:  $modelDir\n")
            File(modelDir).listFiles()?.forEach { diag.append("  ${it.name} ${it.length()}B\n") }
                ?: diag.append("  (models folder empty!)\n")

            android.util.Log.i("PU", diag.toString())

            var success = false
            var exception = ""
            try {
                success = fn(imagePath, outPath, modelDir)
            } catch (e: Throwable) {
                exception = e.toString()
                android.util.Log.e("PU", "Throwable: $exception")
            }

            val outFile = File(outPath)
            diag.append("\nResult:  ${if (success) "✓ SUCCESS" else "✗ FAILED"}\n")
            diag.append("Output exists: ${outFile.exists()}, ${outFile.length()}B\n")
            if (exception.isNotEmpty()) diag.append("Exception: $exception\n")
            if (crashFile.exists()) diag.append("\nCRASH LOG:\n${crashFile.readText()}\n")

            runOnUiThread {
                dlg?.dismiss()
                if (success && outFile.exists() && outFile.length() > 0) {
                    binding.imageView.setImageBitmap(BitmapFactory.decodeFile(outPath))
                    toast("$name completed.")
                    selectedImagePath = outPath
                } else {
                    showDebugDialog("❌ $name Failed — kirim screenshot ini", diag.toString())
                }
            }
        }
    }
}

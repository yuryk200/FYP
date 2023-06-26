package com.example.chemsolve;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;

import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.unity3d.player.UnityPlayerActivity;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Objects;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity
{
    ImageView view;
    Button but, but2;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        view = findViewById(R.id.imageview);
        but = findViewById(R.id.button);
        but2 = findViewById(R.id.button1);

        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
        {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, 100);
        }

        but.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, 100);
            }
        });

        but2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, UnityActivity.class);
                startActivity(intent);
            }
        });

    }

    @Override
    protected  void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 100)
        {
            Bundle extras = data.getExtras();
            Bitmap bitmap = (Bitmap) extras.get("data");
            view.setImageBitmap(bitmap);
            new NetworkTask().execute(bitmap);

        }
    }

    private class NetworkTask extends AsyncTask<Bitmap, Void, String>
    {
        @Override
        protected String doInBackground(Bitmap... bitmaps) {
            Bitmap bitmap = bitmaps[0];
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
            byte[] byteArray = stream.toByteArray();

            OkHttpClient client = new OkHttpClient();
            RequestBody requestBody = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("key", "your-key-value") // add key-value pair here
                    .addFormDataPart("image", "image.png", RequestBody.create(MediaType.parse("image/png"), byteArray))
                    .build();

            Request request = new Request.Builder()
                    .url("http://172.27.232.24:5000/interactive_shell")
                    .post(requestBody)
                    .build();

            String responseBody = null;
            try
            {
                Response response = client.newCall(request).execute();
                responseBody = Objects.requireNonNull(response.body()).string();
                Log.d("responseBody", responseBody);
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }

            return responseBody;
        }

        protected void onPostExecute(String responseBody)
        {
            // Process API response
            if (responseBody != null)
            {
                // Display response in a TextView
                TextView resultTextView = findViewById(R.id.textView);
                resultTextView.setText(responseBody);
            }
            else
            {
                // Display error message in a Toast
                Toast.makeText(MainActivity.this, "An error occurred", Toast.LENGTH_SHORT).show();
            }
        }
    }

}
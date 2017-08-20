package com.imperialsoupgmail.tesseractexample;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;

import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity4 extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }
    }
    private static final String TAG = MainActivity4.class.getCanonicalName();

    private CameraBridgeViewBase mOpenCvCameraView;

    private Mat                    mRgba;
    private Mat                    mIntermediateMat;
    private Mat                    mGray;
    private TessBaseAPI mTess;
    String datapath = "";
    Mat hierarchy;


    List<MatOfPoint> contours;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_calibration_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);
        String language = "eng";
        datapath = getFilesDir()+ "/tesseract/";
        mTess = new TessBaseAPI();

        checkFile(new File(datapath + "tessdata/"));

        mTess.init(datapath, language);


    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this,
                mLoaderCallback);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }



    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        hierarchy = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mIntermediateMat.release();
        hierarchy.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        contours = new ArrayList<MatOfPoint>();
        hierarchy = new Mat();

//        Imgproc.GaussianBlur(mRgba, mRgba, new org.opencv.core.Size(5, 5), 0);
//        Imgproc.Canny(mRgba, mIntermediateMat, 80, 100);
//        Imgproc.findContours(mIntermediateMat, contours, hierarchy, Imgproc.RETR_TREE,
//                Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));
//        for(MatOfPoint contour: contours){
//            if(contour.height() > 100 || contour.width() > 100){
//
//                Rect rect = Imgproc.boundingRect(contour);
//                Mat submat = mRgba.submat(rect);
//                Bitmap bitmap = Bitmap.createBitmap(submat.cols(), submat.rows(), Bitmap.Config.ARGB_8888);
//                Imgproc.putText(mRgba,"YES",new Point(100, 500),4,8,new Scalar(0,255,0,255));
//                Utils.matToBitmap(submat, bitmap);
//                mTess.setImage(bitmap);
//                String OCRresult = mTess.getUTF8Text();
//                Log.d("Main", " HII " + OCRresult);
//            }
//        }
//        hierarchy.release();
//        Imgproc.drawContours(mRgba, contours, -1, new Scalar(255,0,0,255));

        Imgproc.GaussianBlur(mRgba,mIntermediateMat,new Size(9,9),2,2);
        Imgproc.Canny(mRgba, mIntermediateMat, 80, 100);
        Imgproc.findContours(mIntermediateMat, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));
        hierarchy.release();
        for ( int contourIdx=0; contourIdx < contours.size(); contourIdx++ )
        {
            MatOfPoint matOfPoint = contours.get(contourIdx);
            if(matOfPoint.size().area() > 100){
                // Minimum size allowed for consideration
                MatOfPoint2f approxCurve = new MatOfPoint2f();
                MatOfPoint2f contour2f = new MatOfPoint2f( contours.get(contourIdx).toArray() );
                //Processing on mMOP2f1 which is in type MatOfPoint2f
                double approxDistance = Imgproc.arcLength(contour2f, true)*0.02;
                Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

                //Convert back to MatOfPoint
                MatOfPoint points = new MatOfPoint( approxCurve.toArray() );

                // Get bounding rect of contour
                Rect rect = Imgproc.boundingRect(points);

                Imgproc.rectangle(mRgba, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(255, 0, 0, 255), 3);


                Rect rect2 = Imgproc.boundingRect(contours.get(contourIdx));
                Mat submat = mRgba.submat(rect2);
                Bitmap bitmap = Bitmap.createBitmap(submat.cols(), submat.rows(), Bitmap.Config.ARGB_8888);
//                Imgproc.putText(mRgba,"YES",new Point(100, 500),4,8,new Scalar(0,255,0,255));
                Utils.matToBitmap(submat, bitmap);
                mTess.setImage(bitmap);
                String OCRresult = mTess.getUTF8Text();
                Log.d("Main", " HII " + OCRresult);

            }

        }

        return mRgba;
    }

    private void findContours(Mat mat, List<MatOfPoint> contours){
        Mat rgb = new Mat();

        Imgproc.pyrDown(mat, rgb);

        Mat small = new Mat();

        Imgproc.cvtColor(rgb, small, Imgproc.COLOR_RGB2GRAY);

        Mat grad = new Mat();

        Mat morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3,3));

        Imgproc.morphologyEx(small, grad, Imgproc.MORPH_GRADIENT , morphKernel);

        Mat bw = new Mat();

        Imgproc.threshold(grad, bw, 0.0, 255.0, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        Mat connected = new Mat();

        morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1,9));

        Imgproc.morphologyEx(bw, connected, Imgproc.MORPH_CLOSE  , morphKernel);


        Mat mask = Mat.zeros(bw.size(), CvType.CV_8UC1);

        Mat hierarchy = new Mat();

        Imgproc.findContours(connected, contours, hierarchy, Imgproc.RETR_CCOMP,
                Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));

    }

    private void checkFile(File dir) {
        if (!dir.exists()&& dir.mkdirs()){
            copyFiles();
        }
        if(dir.exists()) {
            String datafilepath = datapath+ "/tessdata/eng.traineddata";
            File datafile = new File(datafilepath);

            if (!datafile.exists()) {
                copyFiles();
            }
        }
    }

    private void copyFiles() {
        try {
            String filepath = datapath + "/tessdata/eng.traineddata";
            AssetManager assetManager = getAssets();

            InputStream instream = assetManager.open("tessdata/eng.traineddata");
            OutputStream outstream = new FileOutputStream(filepath);

            byte[] buffer = new byte[1024];
            int read;
            while ((read = instream.read(buffer)) != -1) {
                outstream.write(buffer, 0, read);
            }


            outstream.flush();
            outstream.close();
            instream.close();

            File file = new File(filepath);
            if (!file.exists()) {
                throw new FileNotFoundException();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
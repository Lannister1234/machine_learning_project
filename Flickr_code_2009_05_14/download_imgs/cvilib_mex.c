/*
 *      Coffeeright (c) 2002-2003 by J. Luis
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License as published by
 *      the Free Software Foundation; version 2 of the License.
 *
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *
 *      Contact info: w3.ualg.pt/~jluis
 *--------------------------------------------------------------------*/
/* Program:	cvlib_mex.c
 * Purpose:	matlab callable routine to interface with some OpenCV library functions
 *
 * Revision 1.0  31/08/2006 Joaquim Luis
 * Revision 2.0  19/10/2006 JL	Edge 'laplace' needed exlicit kernel input
 * Revision 3.0  27/10/2006 JL	Updated cvHoughCircles call to 1.0
 * Revision 4.0  07/11/2006 JL	Erode & Diltate in cvHoughCircles  (almost the same shit)
 * Revision 5.0  29/11/2006 JL	Added half a dozen of functions more (line, rect, circ, poly, ellip, inpaint)
 * Revision 6.0  02/12/2006 JL	Added FillPoly and FillConvexPoly
 *
 */

#include <math.h>
#include "mex.h"
#include <cv.h>

#define	TRUE	1
#define	FALSE	0

struct CV_CTRL {
	/* active is TRUE if the option has been activated */
	struct UInt8 {			/* Declare byte pointers */
		int active;
		unsigned char *img_out, *img_in, *tmp_img_in, *tmp_img_out;
	} UInt8;
	struct Int8 {			/* Declare byte pointers */
		int active;
		char *img_out, *img_in, *tmp_img_in, *tmp_img_out;
	} Int8;
	struct UInt16 {			/* Declare short int pointers */
		int active;
		unsigned short int *img_out, *img_in, *tmp_img_in, *tmp_img_out;
	} UInt16;
	struct Int16 {			/* Declare unsigned short int pointers */
		int active;
		short int *img_out, *img_in, *tmp_img_in, *tmp_img_out;
	} Int16;
	struct Int32 {			/* Declare unsigned int pointers */
		int active;
		int *img_out, *img_in, *tmp_img_in, *tmp_img_out;
	} Int32;
	struct Float {			/* Declare float pointers */
		int active;
		float *img_out, *img_in, *tmp_img_in, *tmp_img_out;
	} Float;
	struct Double {			/* Declare double pointers */
		int active;
		double *img_out, *img_in, *tmp_img_in, *tmp_img_out;
	} Double;
};

void JapproxPoly(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);
void Jresize(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);
void Jfloodfill(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);
void JgoodFeatures(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);
void JhoughLines2(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);
void JhoughCircles(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);
void Jedge(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[], const char *method);
void JerodeDilate(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[], const char *method);
void JmorphologyEx(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);
void Jcolor(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);
void Jarithm(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[], const char *op);
void JaddWeighted(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);
void Jflip(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);
void JGetQuadrangleSubPix(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);
void Jfilter(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);
void Jsmooth(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);
void Jegipt(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[], const char *op);
void Jshapes(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[], const char *op);
void Jinpaint(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);

void Set_pt_Ctrl_in (struct CV_CTRL *Ctrl, const mxArray *pi , mxArray *pit, int interl);
void Set_pt_Ctrl_out1 ( struct CV_CTRL *Ctrl, mxArray *pi );
void Set_pt_Ctrl_out2 (struct CV_CTRL *Ctrl, mxArray *po, int interl);
int getNK(const mxArray *p, int which);

void interleave(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir);
void interleaveUI8(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir);
void interleaveI8(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir);
void interleaveUI16(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir);
void interleaveI16(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir);
void interleaveI32(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir);
void interleaveF32(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir);
void interleaveF64(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir);
void interleaveBlind(unsigned char in[], unsigned char out[], int nx, int ny, int nBands, int dir);
void interleaveDouble(double in[], double out[], int nx, int ny);
void localSetData(struct CV_CTRL *Ctrl, IplImage* img, int dir, int step);
void getDataType(struct CV_CTRL *Ctrl, const mxArray *prhs[], int *nBytes, int *img_depth);
void cvResizeUsage(), floodFillUsage(), goodFeaturesUsage(), houghLines2Usage();
void cannyUsage(), sobelUsage(), laplaceUsage(), erodeUsage(), dilateUsage();
void morphologyexUsage(), colorUsage(), flipUsage(), filterUsage(), findContoursUsage();
void JfindContours(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);
void arithmUsage(), addWeightedUsage(), pyrDUsage(), pyrUUsage(), houghCirclesUsage();
void smoothUsage(), lineUsage(), plineUsage(), rectUsage(), circUsage(), eBoxUsage();
void inpaintUsage(), fillConvUsage(), fillPlineUsage();



/* --------------------------------------------------------------------------- */
/* Matlab Gateway routine */
void mexFunction(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	const char *funName;

	if (n_in == 0) {
		mexPrintf("List of currently coded OPENCV functions (original name in parentesis):\n");
		mexPrintf("To get a short online help type cvlib_mex(funname)\n");
		mexPrintf("E.G. cvlib_mex('resize')\n\n");
		mexPrintf("\tadd (cvAdd)\n");
		mexPrintf("\taddS (cvAddS)\n");
		mexPrintf("\taddweighted (cvAddWeighted)\n");
		//mexPrintf("\taffine2 (cvWarpAffine)\n");	// No help yet
		mexPrintf("\tcanny (cvCanny)\n");
		mexPrintf("\tcircle (cvCircle)\n");
		mexPrintf("\tcolor (cvCvtColor)\n");
		mexPrintf("\tcontours (cvFindContours)\n");
		mexPrintf("\tdilate (cvDilate)\n");
		mexPrintf("\tdiv (cvDiv)\n");
		mexPrintf("\teBox (cvEllipseBox)\n");
		mexPrintf("\terode (cvErode)\n");
		mexPrintf("\tfilter (cvFilter2D)\n");
		mexPrintf("\tfillpoly (cvFillPoly)\n");
		mexPrintf("\tflip (cvFlip)\n");
		mexPrintf("\tfloodfill (cvFloodFill)\n");
		mexPrintf("\tgoodfeatures (cvGoodFeaturesToTrack)\n");
		mexPrintf("\thoughlines2 (cvHoughLines2)\n");
		mexPrintf("\thoughcircles (cvHoughCircles)\n");
		mexPrintf("\tinpaint (cvInpaint)\n");
		mexPrintf("\tlaplace (cvLaplace)\n");
		mexPrintf("\tline (cvLine)\n");
		mexPrintf("\tmorphologyex (cvMorphologyEx)\n");
		mexPrintf("\tmul (cvMul)\n");
		mexPrintf("\tpyrD (cvPyrDown)\n");
		mexPrintf("\tpyrU (cvPyrUp)\n");
		mexPrintf("\trectangle (cvRectangle)\n");
		mexPrintf("\tresize (cvResize)\n");
		mexPrintf("\tsmooth (cvSmooth)\n");
		mexPrintf("\tsobel (cvSobel)\n");
		mexPrintf("\tsub (cvSub)\n");
		mexPrintf("\tsubS (cvSubS)\n");
		return;
	}

	if(!mxIsChar(prhs[0]))
		mexErrMsgTxt("CVLIB_MEX: First argument must be a string with the function to call!");
	else
		funName = (char *)mxArrayToString(prhs[0]);

	if (!strcmp(funName,"resize"))
		Jresize(n_out, plhs, n_in, prhs);

	else if (!strcmp(funName,"dp"))
		JapproxPoly(n_out, plhs, n_in, prhs);

	else if (!strcmp(funName,"floodfill"))
		Jfloodfill(n_out, plhs, n_in, prhs);

	else if (!strncmp(funName,"lin",3) || !strncmp(funName,"rec",3) || !strncmp(funName,"cir",3) ||
		!strcmp(funName,"eBox"))
		Jshapes(n_out, plhs, n_in, prhs, funName);


	else if (!strcmp(funName,"goodfeatures"))
		JgoodFeatures(n_out, plhs, n_in, prhs);

	else if (!strcmp(funName,"houghlines2"))
		JhoughLines2(n_out, plhs, n_in, prhs);

	else if (!strcmp(funName,"houghcircles"))
		JhoughCircles(n_out, plhs, n_in, prhs);

	else if (!strcmp(funName,"inpaint"))
		Jinpaint(n_out, plhs, n_in, prhs);

	else if (!strcmp(funName,"contours"))
		JfindContours(n_out, plhs, n_in, prhs);

	else if (!strcmp(funName,"sobel") || !strcmp(funName,"laplace") || !strcmp(funName,"canny"))
		Jedge(n_out, plhs, n_in, prhs, funName);

	else if (!strcmp(funName,"erode") || !strcmp(funName,"dilate"))
		JerodeDilate(n_out, plhs, n_in, prhs, funName);

	else if (!strcmp(funName,"morphologyex"))
		JmorphologyEx(n_out, plhs, n_in, prhs);

	else if (!strcmp(funName,"color"))
		Jcolor(n_out, plhs, n_in, prhs);

	else if (!strcmp(funName,"affine2"))
		JGetQuadrangleSubPix(n_out, plhs, n_in, prhs);

	else if ( !strcmp(funName,"add") || !strcmp(funName,"sub") || !strcmp(funName,"mul") ||
		 !strcmp(funName,"div") || !strcmp(funName,"addS") || !strcmp(funName,"subS") )
		Jarithm(n_out, plhs, n_in, prhs, funName);

	else if (!strcmp(funName,"addweighted"))
		JaddWeighted(n_out, plhs, n_in, prhs);

	else if (!strcmp(funName,"flip"))
		Jflip(n_out, plhs, n_in, prhs);

	else if (!strcmp(funName,"filter"))
		Jfilter(n_out, plhs, n_in, prhs);

	else if (!strcmp(funName,"smooth"))
		Jsmooth(n_out, plhs, n_in, prhs);

	else if (!strcmp(funName,"pyrU") || !strcmp(funName,"pyrD") )
		Jegipt(n_out, plhs, n_in, prhs, funName);

	else
		mexErrMsgTxt("CVLIB_MEX: unrecognized function name!");
}


/* --------------------------------------------------------------------------- */
void Jresize(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	int nx, ny, nBands, out_dims[3], nx_out, ny_out, nBytes, isMN, img_depth;
	int cv_code = CV_INTER_LINEAR;
	double	*ptr_d, size_fac;
	const char *interpMethod;
	IplImage* src_img = 0;
	IplImage* dst_img = 0;
	mxArray *ptr_in, *ptr_out;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for errors in user's call to function.  -------------------- */
	if (n_in == 1) { cvResizeUsage(); return; }
	if (n_out > 2)
		mexErrMsgTxt("RESIZE returns only one argument!");
	if (n_in < 3 || n_in > 4 )
		mexErrMsgTxt("RESIZE needs two or three input arguments!");

	ptr_d = mxGetPr(prhs[2]);
	if (mxGetM(prhs[2]) * mxGetN(prhs[2]) == 2) {
		ny_out = (int)ptr_d[0];		nx_out = (int)ptr_d[1];
		isMN = 1;
	}
	else if (mxGetM(prhs[2]) * mxGetN(prhs[2]) == 1) {
		size_fac = *ptr_d;
		isMN = 0;
	}
	else
		mexErrMsgTxt("RESIZE: second argument must be a scalar or a two elements vector!");

	if (n_in == 4) {
		if(!mxIsChar(prhs[3]))
			mexErrMsgTxt("RESIZE: Third argument must be a valid string!");
		else
			interpMethod = (char *)mxArrayToString(prhs[3]);
		if (!strcmp(interpMethod,"nearest"))
			cv_code = CV_INTER_NN;
		else if (!strcmp(interpMethod,"bilinear"))
			cv_code = CV_INTER_LINEAR;
		else if (!strcmp(interpMethod,"bicubic"))
			cv_code = CV_INTER_CUBIC;
		else if (!strcmp(interpMethod,"area"))
			cv_code = CV_INTER_AREA;
		else
			mexErrMsgTxt("RESIZE: Unknow interpolation method!");
	}

	if ( cv_code != CV_INTER_NN && !(mxIsLogical(prhs[1]) || mxIsUint8(prhs[1]) ||
		 mxIsUint16(prhs[1]) || mxIsSingle(prhs[1])) ) {
		mexPrintf("RESIZE ERROR: Interpolation methods other than nearest-neighbor accept only\n");
		mexErrMsgTxt("           the folloowing input data types: logical, uint8, uint16 & single.\n");
	}

	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	if (!isMN) {
		nx_out = cvRound(nx * size_fac);
		ny_out = cvRound(ny * size_fac);
	}
	out_dims[0] = ny_out;
	out_dims[1] = nx_out;
	out_dims[2] = nBands;
	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointers for output and temporary arrays ------------------------ */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	ptr_out = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  out_dims, mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */
	src_img = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, src_img, 1, nx * nBands * nBytes );

	Set_pt_Ctrl_out1 ( Ctrl, ptr_out ); 
	//dst_img = cvCreateImage(cvSize(nx_out, ny_out), img_depth , nBands );
	dst_img = cvCreateImageHeader(cvSize(nx_out, ny_out), img_depth , nBands );
	localSetData( Ctrl, dst_img, 2, nx_out * nBands * nBytes );

	cvResize(src_img,dst_img,cv_code);

	cvReleaseImageHeader( &src_img );
	mxDestroyArray(ptr_in);

	plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  out_dims, mxGetClassID(prhs[1]), mxREAL);
	Set_pt_Ctrl_out2 ( Ctrl, plhs[0], 1 ); 		/* Set pointer & desinterleave */

	cvReleaseImageHeader( &dst_img ); //cvCreateImage is ued to initialize it, why not deallocate it accordingly?
	//cvReleaseImage( &dst_img );
	mxDestroyArray(ptr_out); 
	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}

/* --------------------------------------------------------------------------- */
void Jfloodfill(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	unsigned char *mask_img, *tmp_mask;
	int nx, ny, nx2, ny2, nBands, m, n, nx_var, c = 0, nBytes, img_depth;

	int lo, up, flags, r, g, b, x, y;
	int ffill_case = 1;
	int lo_diff = 20, up_diff;
	int connectivity = 4;
	int is_mask = 0;
	int new_mask_val = 255;
	double *ptr_d;
	IplImage* src_img = 0;
	IplImage* mask = 0;
	CvPoint seed;
	CvConnectedComp comp;
	CvScalar color;
	CvScalar brightness;
	CvSize roi = {0,0};
	mxArray *ptr_in, *mx_ptr;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for errors in user's call to function.  -------------------- */
	if (n_in == 1) { floodFillUsage(); return; }
	else if (n_out > 2 || n_out == 0)
		mexErrMsgTxt("FLOODFILL returns one or two output arguments!");
	else if (n_out == 2)
		is_mask = 1;


	/* Check that input image is of type UInt8 */
	if (!mxIsUint8(prhs[1]))
		mexErrMsgTxt("FLOODFILL ERROR: Invalid input data type. Only valid type is: UInt8.\n");

	if (n_in != 3)
		mexErrMsgTxt("FLOODFILL requires 2 input arguments!");

	else if (mxIsStruct(prhs[2])) {
		mx_ptr = mxGetField(prhs[2], 0, "Point");
		if (mx_ptr == NULL)
			mexErrMsgTxt("FLOODFILL 'Point' field not provided");
		ptr_d = mxGetPr(mx_ptr);
		x = (int)ptr_d[0];	y = (int)ptr_d[1];

		mx_ptr = mxGetField(prhs[2], 0, "Tolerance");
		if (mx_ptr != NULL) {
			ptr_d = mxGetPr(mx_ptr);
			lo_diff = (int)ptr_d[0];
		}

		mx_ptr = mxGetField(prhs[2], 0, "Connect");
		if (mx_ptr != NULL) {
			ptr_d = mxGetPr(mx_ptr);
			connectivity = (int)ptr_d[0];
		}

		mx_ptr = mxGetField(prhs[2], 0, "FillColor");
		if (mx_ptr != NULL) {
			ptr_d = mxGetPr(mx_ptr);
			r = (int)ptr_d[0];
			g = (int)ptr_d[1];
			b = (int)ptr_d[2];
		}
		else {
			b = rand() & 255, g = rand() & 255, r = rand() & 255;
		}
	}
	else {		/* cvfill_mex(img,[x y]) mode */
		if (mxGetM(prhs[2]) * mxGetN(prhs[2]) != 2)
			mexErrMsgTxt("FLOODFILL: Seed point error. Must be a 2 elements vector.");
		ptr_d = mxGetPr(prhs[2]);
		x = (int)ptr_d[0];		y = (int)ptr_d[1];
		b = rand() & 255, g = rand() & 255, r = rand() & 255;
	}
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	up_diff = lo_diff;	/* Don't see any reason to have them different */

	/* ------ Create pointer for temporary array ------------------------------------- */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		 mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */

	src_img = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, src_img, 1, nx * nBands * nBytes );

	if( is_mask ) {
		nx2 = nx + 2;	ny2 = ny + 2;
		mask = cvCreateImage( cvSize(nx2, ny2), IPL_DEPTH_8U, 1 );
		cvZero( mask );
	}

	seed = cvPoint(x,y);
	lo = ffill_case == 0 ? 0 : lo_diff;
	up = ffill_case == 0 ? 0 : up_diff;
	flags = connectivity + (new_mask_val << 8) + (ffill_case == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);

	if( nBands == 3 ) {
		color = CV_RGB( r, g, b );
		cvFloodFill( src_img, seed, color, CV_RGB( lo, lo, lo ),
			CV_RGB( up, up, up ), &comp, flags, is_mask ? mask : NULL );
	}
	else {
		brightness = cvRealScalar((r*2 + g*7 + b + 5)/10);
		cvFloodFill( src_img, seed, brightness, cvRealScalar(lo),
			cvRealScalar(up), &comp, flags, is_mask ? mask : NULL );
	}

	plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		 mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);

	/* desinterleave */
	interleaveBlind (Ctrl->UInt8.tmp_img_in, (unsigned char *)mxGetData(plhs[0]), nx, ny, nBands, -1);

	cvReleaseImageHeader( &src_img );
	mxDestroyArray(ptr_in);
	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */

	if ( is_mask ) {
		nx_var = mask->widthStep;	/* BUG, this value is changed inside cvFloodFill */
		tmp_mask = (unsigned char *)mxCalloc((nx*ny),sizeof(char));

		/* Crop the mask such that it will have the same size as original image */
		for (m = 1, c = 0; m < ny+1; m++)
			for (n = 1; n < nx+1; n++)
				tmp_mask[c++] = (unsigned char)(mask->imageData[m*nx_var + n] & 1); 

		plhs[1] = mxCreateNumericMatrix(ny, nx, mxLOGICAL_CLASS, mxREAL);
		mask_img = mxGetData(plhs[1]);

		interleaveBlind (tmp_mask, mask_img, nx, ny, 1, -1); /* Change from C to ML order */
		mxFree((void *)tmp_mask);
		cvReleaseImage( &mask );
	}
}


/* --------------------------------------------------------------------------- */
void Jshapes(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[], const char *method) {
	int nx, ny, nBands, nBytes, img_depth, inplace = FALSE;
	int r, g, b, radius = 0, thickness = 1, line_type = 8;
	int ind_opt = 4;	/* Index of first optional argument in prhs */
	double *ptr_d;
	IplImage *src_img = 0, *dst = 0;
	CvPoint pt1, pt2;
	CvScalar color;
	CvBox2D	box;
	mxArray *ptr_in, *mx_ptr;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for input and errors in user's call to function. ----------------- */
	if (n_in == 1) { 
		if (!strncmp(method,"lin",3)) lineUsage();
		else if (!strncmp(method,"rec",3)) rectUsage();
		else if (!strncmp(method,"cir",3)) circUsage();
		else if (!strcmp(method,"eBox")) eBoxUsage();
		return;
	}
	else if (n_out > 1 )
		mexErrMsgTxt("SHAPES returns either one or zero arguments!");

	/* Check that input image is of type UInt8 */
	if (!mxIsUint8(prhs[1]))
		mexErrMsgTxt("SHAPES ERROR: Invalid input data type. Only valid type is: UInt8.\n");

	if (!strcmp(method,"eBox") && n_in < 3)
		mexErrMsgTxt("EllipseBox requires at least 2 input arguments!");
	else if (strcmp(method,"eBox") && n_in < 4)
		mexErrMsgTxt("SHAPES requires at least 3 input arguments!");

	if (!strncmp(method,"lin",3) || !strncmp(method,"rec",3)) {
		/* Those are mandatory */
		if (mxGetM(prhs[2]) * mxGetN(prhs[2]) != 2)
			mexErrMsgTxt("SHAPES: First point error. Must be a 2 elements vector.");
		if (mxGetM(prhs[3]) * mxGetN(prhs[3]) != 2)
			mexErrMsgTxt("SHAPES: Second point error. Must be a 2 elements vector.");

		/* OK, now read the two points and assign them to the cvPoint structs */
		ptr_d = (double *)mxGetData(prhs[2]);
		pt1.x = (int)ptr_d[0];		pt1.y = (int)ptr_d[1];
		ptr_d = (double *)mxGetData(prhs[3]);
		pt2.x = (int)ptr_d[0];		pt2.y = (int)ptr_d[1];
	}
	else if (!strncmp(method,"cir",3)) {
		if (mxGetM(prhs[2]) * mxGetN(prhs[2]) != 2)
			mexErrMsgTxt("CIRCLE: First point error - the CENTER. Must be a 2 elements vector.");
		ptr_d = (double *)mxGetData(prhs[2]);
		pt1.x = (int)ptr_d[0];		pt1.y = (int)ptr_d[1];
		ptr_d = (double *)mxGetData(prhs[3]);
		radius = (int)ptr_d[0];
	}
	else if (!strcmp(method,"eBox")) {
		if (mxIsStruct(prhs[2])) {
			mx_ptr = mxGetField(prhs[2], 0, "center");
			if (mx_ptr == NULL)
				mexErrMsgTxt("EllipseBox 'center' field not provided");
			if (mxGetM(mxGetField(prhs[2],0,"center")) * mxGetN(mxGetField(prhs[2],0,"center")) != 2)
				mexErrMsgTxt("EllipseBox: 'center' must contain a 2 elements vector");
			ptr_d = mxGetPr(mx_ptr);
			box.center.x = (float)ptr_d[0];
			box.center.y = (float)ptr_d[1];

			mx_ptr = mxGetField(prhs[2], 0, "size");
			if (ptr_d == NULL)
				mexErrMsgTxt("EllipseBox 'size' field not provided");
			if (mxGetM(mxGetField(prhs[2],0,"size")) * mxGetN(mxGetField(prhs[2],0,"size")) != 2)
				mexErrMsgTxt("EllipseBox: 'size' must contain a 2 elements vector");
			ptr_d = mxGetPr(mx_ptr);
			box.size.width = (float)ptr_d[1];	/* On purpose change of width & height. The man is again */
			box.size.height = (float)ptr_d[0];	/* very confuse. I think they mixed up one and the other */

			mx_ptr = mxGetField(prhs[2], 0, "angle");
			if (mx_ptr == NULL)
				box.angle = 0.0f;
			else {
				ptr_d = mxGetPr(mx_ptr);
				box.angle = (float)ptr_d[0];
			}

			n_in++;		/* Since cvEllipseBox has one less input arg than cvLine, etc */
			ind_opt = 3;
		}
		else
			mexErrMsgTxt("EllipseBox: Second argument must be a structure.");
	}

	color = cvScalarAll(255);	/* Default to a white line */

	if (n_in > 4 && !mxIsEmpty(prhs[ind_opt])) {			/* Line color */
		ptr_d = (double *)mxGetData(prhs[ind_opt]);
		if (mxGetM(prhs[ind_opt]) * mxGetN(prhs[ind_opt]) == 1) {	/* Gray line */
			r = (int)ptr_d[0];
			color = CV_RGB( r, r, r );
		}
		else if (mxGetM(prhs[ind_opt]) * mxGetN(prhs[ind_opt]) == 3) {	/* Color line */
			r = (int)ptr_d[0];	g = (int)ptr_d[1];	b = (int)ptr_d[2];
			color = CV_RGB( r, g, b );
		}
		else
			mexErrMsgTxt("LINE: Fourth argument must be a 1 or a 3 elements vector.");

		ind_opt++;
	}
	if (n_in > 5 && !mxIsEmpty(prhs[ind_opt]))			/* Line thickness */
		thickness = (int)(*mxGetPr(prhs[ind_opt++]));
	if (n_in > 6 && !mxIsEmpty(prhs[ind_opt]))			/* Line type */
		line_type = (int)(*mxGetPr(prhs[ind_opt]));

	if (thickness < 0) thickness = -1;		/* Bug in OpenCV */

	if (n_out == 0)
		inplace = TRUE;
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointer for temporary array ------------------------------------- */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		 mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */

	src_img = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, src_img, 1, nx * nBands * nBytes );

	if (!inplace) {
		plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
		dst = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
		cvSetImageData( dst, (void *)mxGetData(plhs[0]), nx * nBytes * nBands );
		localSetData( Ctrl, dst, 1, nx * nBands * nBytes );
		if (!strncmp(method,"lin",3))
			cvLine( dst, pt1, pt2, color, thickness, line_type, 0 );
		else if (!strncmp(method,"rec",3))
			cvRectangle( dst, pt1, pt2, color, thickness, line_type, 0 );
		else if (!strncmp(method,"cir",3))
			cvCircle( dst, pt1, radius, color, thickness, line_type, 0 );
		else if (!strcmp(method,"eBox")) 
			cvEllipseBox( dst, box, color, thickness, line_type, 0 );
		interleaveBlind (Ctrl->UInt8.tmp_img_in, (unsigned char *)mxGetData(plhs[0]), nx, ny, nBands, -1);
		cvReleaseImageHeader( &dst );
	}
	else {
		if (!strncmp(method,"lin",3))
			cvLine( src_img, pt1, pt2, color, thickness, line_type, 0 );
		else if (!strncmp(method,"rec",3))
			cvRectangle( src_img, pt1, pt2, color, thickness, line_type, 0 );
		else if (!strncmp(method,"cir",3))
			cvCircle( src_img, pt1, radius, color, thickness, line_type, 0 );
		else if (!strcmp(method,"eBox")) 
			cvEllipseBox( src_img, box, color, thickness, line_type, 0 );
		/* desinterleave */
		interleaveBlind (Ctrl->UInt8.tmp_img_in, (unsigned char *)mxGetData(prhs[1]), nx, ny, nBands, -1);
	}

	cvReleaseImageHeader( &src_img );
	mxDestroyArray(ptr_in);
	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}

/* --------------------------------------------------------------------------- */
void JgoodFeatures(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	const int max_features = 100000;
	int nx, ny, n, i, nBytes, img_depth, nBands, corner_count, nOut_corners = max_features;
	unsigned char *ptr_gray;

	double *ptr_d, quality_level = 0.1, min_distance = 10;
	IplImage *src_img = 0, *eig_image, *temp_image, *src_gray;
	CvSize roi = {0,0};
	CvPoint2D32f* corners;
	mxArray *ptr_in;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for errors in user's call to function.  -------------------- */
	if (n_in == 1) { goodFeaturesUsage(); return; }
	/* Check that input image is of type UInt8 */
	if ( !mxIsUint8(prhs[1]) )
		mexErrMsgTxt("GOODFEATURES: Invalid input data type. Valid type is: UInt8.\n");
	if (n_out > 1)
		mexErrMsgTxt("GOODFEATURES returns only one output argument!");

	if (n_in >= 3 && !mxIsEmpty(prhs[2]))
		nOut_corners = (int)(*mxGetPr(prhs[2]));
	if (n_in >= 4 && !mxIsEmpty(prhs[3]))
		quality_level = *mxGetPr(prhs[3]);
	if (n_in >= 5 && !mxIsEmpty(prhs[4]))
		min_distance = *mxGetPr(prhs[4]);
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointer for temporary array ------------------------------------- */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */

	src_img = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, src_img, 1, nx * nBands * nBytes );

	if (nBands == 3) {			/* Convert to GRAY */
		src_gray = cvCreateImageHeader( cvSize(nx, ny), 8, 1 );
		ptr_gray = (unsigned char *)mxCalloc (nx*ny, sizeof (unsigned char));
		cvSetImageData( src_gray, (void *)ptr_gray, nx );
		cvCvtColor(src_img, src_gray, CV_BGR2GRAY);
		for (i = 0; i < nx*ny; i++)	/* Copy the transformed image into Ctrl field */
			Ctrl->UInt8.tmp_img_in[i] = ptr_gray[i]; 
		mxFree(ptr_gray);
		cvReleaseImageHeader( &src_gray );

		/* Here we're going to cheat the src_img in order to pretend that its 2D */
		src_img->nChannels = 1;
		src_img->widthStep = nx * nBytes;
		src_img->imageSize = ny * src_img->widthStep;
	}

	eig_image = cvCreateImage( cvSize(nx, ny), 32, 1 );
	temp_image = cvCreateImage( cvSize(nx, ny), 32, 1 );

	corners = (CvPoint2D32f*)cvAlloc(max_features*sizeof(corners));

	quality_level = 0.1; 
	corner_count  = max_features; 
	cvGoodFeaturesToTrack( src_img, eig_image, temp_image, corners,
		&corner_count, quality_level, min_distance, NULL, 3, 0, 0.04 );

	/*if (corner_count > 0 )
		cvFindCornerSubPix( src_img, corners, corner_count,
			cvSize(10,10), cvSize(-1,-1),
			cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03));*/

	cvReleaseImage( &eig_image );
	cvReleaseImage( &temp_image );
	cvReleaseImageHeader( &src_img );
	mxDestroyArray(ptr_in);

	/* ------ GET OUTPUT DATA --------------------------- */ 
	nOut_corners = MIN(nOut_corners,corner_count);
	plhs[0] = mxCreateNumericMatrix(nOut_corners, 2, mxDOUBLE_CLASS, mxREAL);
	ptr_d = mxGetPr(plhs[0]);	
	for ( n = 0; n < nOut_corners; n++ ) {
		ptr_d[n] 	      = corners[n].x + 1;	/* +1 because ML is one-based */
		ptr_d[n+nOut_corners] = corners[n].y + 1;
	}
	cvFree((void **)&corners );
	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}

/* --------------------------------------------------------------------------- */
void JhoughLines2(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	int nx, ny, i, nBytes, img_depth, nBands, np, method = CV_HOUGH_PROBABILISTIC, thresh = 50;
	unsigned char *ptr_gray;
	const char *method_s;

	double *ptr_d, *lineS, rho = 1, theta = CV_PI/180, par1 = 50, par2 = 15, a, b, x0, y0;
	IplImage *src_img = 0, *dst_img, *src_gray;
	CvPoint *line;
        CvSeq* lines = 0;
        CvMemStorage* storage;
	mxArray *ptr_in, *ptr_out, *mx_ptr;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for input and errors in user's call to function. ----------------- */
	if (n_in == 1) { houghLines2Usage(); return; }
	/* Check that input image is of type UInt8 */
	if ( !(mxIsUint8(prhs[1]) || mxIsLogical(prhs[1])) )
		mexErrMsgTxt("HOUGHLINES2: Invalid input data type. Valid types are: UInt8 OR Logical.\n");

	if (n_out != 1)
		mexErrMsgTxt("HOUGHLINES2 returns one (and one only) output!");

	if (n_in > 2) {
		if(!mxIsChar(prhs[2]))
			 mexErrMsgTxt("CVLIB_MEX: Third argument must contain the METHOD string!");
		else
			method_s = (char *)mxArrayToString(prhs[2]);
		if (strcmp(method_s,"standard") && strcmp(method_s,"probabilistic"))
			mexErrMsgTxt("CVLIB_MEX: Unknown METHOD!");
		if (!strcmp(method_s,"standard"))
			method = CV_HOUGH_STANDARD;
		if (n_in == 8) {		/* We don't do any error tests in this case */
			ptr_d = (double *)(mxGetData(prhs[3]));		rho = ptr_d[0];
			ptr_d = (double *)(mxGetData(prhs[4])); 	theta = ptr_d[0];
			ptr_d = (double *)(mxGetData(prhs[5]));		thresh = (int)ptr_d[0];
			ptr_d = (double *)(mxGetData(prhs[6]));		par1 = ptr_d[0];
			ptr_d = (double *)(mxGetData(prhs[7]));		par2 = ptr_d[0];
		}
	}
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointers for output and temporary arrays ---------- */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	if (!mxIsLogical(prhs[1]))
		ptr_out = mxCreateNumericMatrix(ny, nx, mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */

	src_img = cvCreateImageHeader( cvSize(nx, ny), 8, nBands );
	localSetData( Ctrl, src_img, 1, nx * nBands * nBytes );

	if (nBands == 3) {			/* Convert to GRAY */
		src_gray = cvCreateImageHeader( cvSize(nx, ny), 8, 1 );
		ptr_gray = (unsigned char *)mxCalloc (nx*ny, sizeof (unsigned char));
		cvSetImageData( src_gray, (void *)ptr_gray, nx );
		cvCvtColor(src_img, src_gray, CV_BGR2GRAY);
		for (i = 0; i < nx*ny; i++)	/* Copy the transformed image into Ctrl field */
			Ctrl->UInt8.tmp_img_in[i] = ptr_gray[i]; 
		mxFree(ptr_gray);
		cvReleaseImageHeader( &src_gray );

		/* Here we're going to cheat the src_img in order to pretend that its 2D */
		src_img->nChannels = 1;
		src_img->widthStep = nx * nBytes;
		src_img->imageSize = ny * src_img->widthStep;
	}

       	storage = cvCreateMemStorage(0);

	if (!mxIsLogical(prhs[1])) {
		dst_img = cvCreateImageHeader( cvSize(nx, ny), 8, 1 );
		Set_pt_Ctrl_out1 ( Ctrl, ptr_out ); 
		localSetData( Ctrl, dst_img, 2, nx * nBytes );

        	cvCanny( src_img, dst_img, 50, 200, 3 );

		lines = cvHoughLines2( dst_img, storage, method, rho, theta, thresh, par1, par2 );
		cvReleaseImageHeader( &dst_img );
		mxDestroyArray(ptr_out);
	}
	else 		/* Input was already a mask array */
		lines = cvHoughLines2( src_img, storage, method, rho, theta, thresh, par1, par2 );

	cvReleaseImageHeader( &src_img );
	mxDestroyArray(ptr_in);

	/* ------ GET OUTPUT DATA --------------------------- */ 
	np = lines->total;		/* total number of pairs of points */

	plhs[0] = mxCreateCellMatrix(np, 1);
	mx_ptr = mxCreateNumericMatrix(2, 2, mxDOUBLE_CLASS, mxREAL);
	ptr_d = mxGetPr(mx_ptr);
	if (method == CV_HOUGH_PROBABILISTIC) {
		for( i = 0; i < np; i++ ) {
			line = (CvPoint *)cvGetSeqElem(lines,i);
			ptr_d[0] = (double)line[0].y + 1;	/* +1 because ML is one-based */
			ptr_d[1] = (double)line[1].y + 1;
			ptr_d[2] = (double)line[0].x + 1;
			ptr_d[3] = (double)line[1].x + 1;
			mxSetCell(plhs[0],i,mxDuplicateArray(mx_ptr));
		}
	}
	else {		/* standard --> STUPID OUTPUT - NOT WORKING */
		for( i = 0; i < np; i++ ) {
			lineS = (double *)cvGetSeqElem(lines,i);
			a = cos(lineS[1]), b = sin(lineS[1]); 
			x0 = a*lineS[0], y0 = b*lineS[0]; 
			//mexPrintf("Merda i=%d\t%f \t%f \tx0=%f \ty0=%f\n",i,lineS[0],lineS[1],x0,y0);
			ptr_d[0] = cvRound(y0 + 1000*(a));
			ptr_d[1] = cvRound(y0 - 1000*(a)); 
			ptr_d[2] = cvRound(x0 + 1000*(-b)); 
			ptr_d[3] = cvRound(x0 - 1000*(-b));
			mxSetCell(plhs[0],i,mxDuplicateArray(mx_ptr));
		}
	}
	mxDestroyArray(mx_ptr);

	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}

/* --------------------------------------------------------------------------- */
void JhoughCircles(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	int nx, ny, i, nBytes, img_depth, nBands, np, thresh = 50, min_radius = 5, max_radius=0;
	unsigned char *ptr_gray;

	float	*circ;
	double	*ptr_d, dp = 1, min_dist = 20, par1 = 50, par2 = 60;
	IplImage *src_img = 0, *src_gray;
	IplConvKernel *element = 0;
        CvSeq* circles = 0;
        CvMemStorage* storage;
	mxArray *ptr_in;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for input and errors in user's call to function. ----------------- */
	if (n_in == 1) { houghCirclesUsage(); return; }
	/* Check that input image is of type UInt8 */
	if ( !(mxIsUint8(prhs[1]) || mxIsLogical(prhs[1])) )
		mexErrMsgTxt("HOUGHCIRCLES: Invalid input data type. Valid types are: Logical or UInt8.\n");

	if (n_out != 1)
		mexErrMsgTxt("HOUGHCIRCLES returns one (and one only) output!");

	if (n_in >= 3 && !mxIsEmpty(prhs[2]))
		dp = *mxGetPr(prhs[2]);
	if (n_in >= 4 && !mxIsEmpty(prhs[3]))
		min_dist = *mxGetPr(prhs[3]);
	if (n_in >= 5 && !mxIsEmpty(prhs[4]))
		par1 = *mxGetPr(prhs[4]);
	if (n_in >= 6 && !mxIsEmpty(prhs[5]))
		par2 = *mxGetPr(prhs[5]);
	if (n_in >= 7 && !mxIsEmpty(prhs[6]))
		min_radius = (int)*mxGetPr(prhs[6]);
	if (n_in >= 8 && !mxIsEmpty(prhs[7]))
		max_radius = (int)*mxGetPr(prhs[7]);
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointer for temporary arrays ------------------------------------ */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */

	src_img = cvCreateImageHeader( cvSize(nx, ny), 8, nBands );
	localSetData( Ctrl, src_img, 1, nx * nBands * nBytes );

	/* Don't really know hoe effective the next commands are, but at least it worked ONCE */
       	element = cvCreateStructuringElementEx( 4*2+1, 4*2+1, 4, 4, CV_SHAPE_ELLIPSE, 0 );
	cvErode(src_img,src_img,element,1);
	cvDilate(src_img,src_img,element,1);

	if (nBands == 3) {			/* Convert to GRAY */
		src_gray = cvCreateImageHeader( cvSize(nx, ny), 8, 1 );
		ptr_gray = (unsigned char *)mxCalloc (nx*ny, sizeof (unsigned char));
		cvSetImageData( src_gray, (void *)ptr_gray, nx );
		cvCvtColor(src_img, src_gray, CV_BGR2GRAY);
		for (i = 0; i < nx*ny; i++)	/* Copy the transformed image into Ctrl field */
			Ctrl->UInt8.tmp_img_in[i] = ptr_gray[i]; 
		mxFree(ptr_gray);
		cvReleaseImageHeader( &src_gray );

		/* Here we're going to cheat the src_img in order to pretend that its 2D */
		src_img->nChannels = 1;
		src_img->widthStep = nx * nBytes;
		src_img->imageSize = ny * src_img->widthStep;
	}

       	storage = cvCreateMemStorage(0);

	/* smooth it, otherwise a lot of false circles may be detected */
	cvSmooth( src_img, src_img, CV_GAUSSIAN, 5, 5, 0, 0 );
	circles = cvHoughCircles( src_img, storage, CV_HOUGH_GRADIENT, dp, min_dist,
				 par1, par2, min_radius, max_radius );

	cvReleaseImageHeader( &src_img );
	mxDestroyArray(ptr_in);

	/* ------------------- GET OUTPUT DATA --------------------------- */ 
	np = circles->total;		/* total number of pairs of points */

	plhs[0] = mxCreateNumericMatrix(np, 3, mxDOUBLE_CLASS, mxREAL);
	ptr_d = mxGetPr(plhs[0]);
	for( i = 0; i < np; i++ ) {
		circ = (float *)cvGetSeqElem(circles,i);
		ptr_d[i]      = (double)circ[0] + 1;	/* +1 because ML is one-based */
		ptr_d[i+np]   = (double)circ[1] + 1;
		ptr_d[i+2*np] = (double)circ[2];
	}

	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}

/* --------------------------------------------------------------------------- */
void JfindContours(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	int nx, ny, i, j, nBytes, img_depth, nBands, np, ncont = 0;
	unsigned char *ptr_gray;

	double *ptr_d;
	IplImage *src_img = 0, *dst_img, *src_gray;
	CvPoint *PointArray;
        CvSeq *contours = 0, *cont = 0;
        CvMemStorage* storage;
	mxArray *ptr_in, *ptr_out, *mx_ptr;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for input and errors in user's call to function. ----------------- */
	if (n_in == 1) { findContoursUsage(); return; }
	/* Check that input image is of type UInt8 */
	if ( !(mxIsUint8(prhs[1]) || mxIsLogical(prhs[1])) )
		mexErrMsgTxt("FINDCONTOURS: Invalid input data type. Valid types are: UInt8 OR Logical.\n");

	if (n_out != 1)
		mexErrMsgTxt("FINDCONTOURS returns one (and one only) output!");
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointers for output and temporary arrays ---------- */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	if (!mxIsLogical(prhs[1]))
		ptr_out = mxCreateNumericMatrix(ny, nx, mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */

	src_img = cvCreateImageHeader( cvSize(nx, ny), 8, nBands );
	localSetData( Ctrl, src_img, 1, nx * nBands * nBytes );

	/*
	if (nBands == 3) {			// Convert to GRAY
		mexPrintf("Converting image to gray in cvlib_mex\n");
		src_gray = cvCreateImageHeader( cvSize(nx, ny), 8, 1 );
		ptr_gray = (unsigned char *)mxCalloc (nx*ny, sizeof (unsigned char));
		cvSetImageData( src_gray, (void *)ptr_gray, nx );
		cvCvtColor(src_img, src_gray, CV_BGR2GRAY);
		for (i = 0; i < nx*ny; i++)	// Copy the transformed image into Ctrl field
			Ctrl->UInt8.tmp_img_in[i] = ptr_gray[i]; 
		mxFree(ptr_gray);
		cvReleaseImageHeader( &src_gray );

		// Here we're going to cheat the src_img in order to pretend that its 2D
		src_img->nChannels = 1;
		src_img->widthStep = nx * nBytes;
		src_img->imageSize = ny * src_img->widthStep;
	}
	*/
       	storage = cvCreateMemStorage(0);
	/*
	if (!mxIsLogical(prhs[1])) {
		mexPrintf("Input image is not logical, running canny\n");
		dst_img = cvCreateImageHeader( cvSize(nx, ny), 8, 1 );
		Set_pt_Ctrl_out1 ( Ctrl, ptr_out ); 
		localSetData( Ctrl, dst_img, 2, nx * nBytes );

        	cvCanny( src_img, dst_img, 50, 200, 3 );

		ncont = cvFindContours( dst_img, storage, &contours, sizeof(CvContour),
				CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cvPoint(1,1) );
		cvReleaseImageHeader( &dst_img );
		mxDestroyArray(ptr_out);
	}
	else{ 
		*/
		/* Input was already a mask array */
		//cvFindContours( src_img, storage, &contours, sizeof(CvContour),
		//		CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );
		//mexPrintf("Input image is logical\n");
	    ncont = cvFindContours( src_img, storage, &contours, sizeof(CvContour),
				CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cvPoint(1,1) ); //jhhays, changed to 1,1 offset
	//}

	//mexPrintf("Merda2 ncont = %d\n", ncont);

	cvReleaseImageHeader( &src_img );  //why only releasing the header?  because that's all that was created
	mxDestroyArray(ptr_in);

	/* ------ GET OUTPUT DATA --------------------------- */ 
	//for(ncont = 0; contours = contours->h_next; ncont++);	/* count number of contours */
    //mexPrintf("Merda2 ncont = %d\n", ncont);
	//for(i = 0; i < 2; contours = contours->h_prev);	/* rewind the contours sequence - STUPID, but with this CV manual ... */
	
	//i = 0;
	//while(contours = contours->h_prev){
	//	mexPrintf("back to node = %d\n", ncont - i);
	//	i++;
	//}

	//ncont = 523;

	plhs[0] = mxCreateCellMatrix(ncont, 1);
	for( i = 0; i < ncont; i++ ) {
		np = contours->total;		/* This is the number points in contour */
		PointArray = (CvPoint*)malloc( np*sizeof(CvPoint) );	/* Alloc memory for contour point set */
		cvCvtSeqToArray(contours, PointArray, CV_WHOLE_SEQ);	/* Get contour point set. */
		mx_ptr = mxCreateNumericMatrix(np, 2, mxDOUBLE_CLASS, mxREAL);
		ptr_d = mxGetPr(mx_ptr);
		for( j = 0; j < np; j++ ) {
			ptr_d[j] = (double)PointArray[j].y;
			ptr_d[j+np] = (double)PointArray[j].x;
		}
 
		mxSetCell(plhs[0],i,mxDuplicateArray(mx_ptr));
		mxDestroyArray(mx_ptr);   //this seems ok, but the memory leak still exists
		free(PointArray);
		contours = contours->h_next;
	}

	//mxDestroyArray(mx_ptr);
	cvReleaseMemStorage(&storage);
	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}

/* --------------------------------------------------------------------------- */
void JapproxPoly(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	int nx, ny, i, j, nBytes, nz, np, ncont = 0;

	double *ptr_d, par1 = 3;
        CvMat *map_matrix = 0, *cont = 0;
	CvMemStorage* storage = cvCreateMemStorage(0);
	mxArray *ptr_in, *ptr_out, *mx_ptr;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for input and errors in user's call to function. ------- */
	if (n_in == 1) { findContoursUsage(); return; }
	ptr_d = mxGetPr(prhs[1]);
	if (n_in == 3)
		par1 = *mxGetPr(prhs[2]);
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nz = getNK(prhs[1],2);

	map_matrix = cvCreateMatHeader( ny, nx, CV_64FC1 );
	cvSetData( map_matrix, (void *)ptr_d, nx*8 );

	cont = cvApproxPoly( map_matrix, sizeof(CvMat), storage, CV_POLY_APPROX_DP, par1, 0 );
return;

	/* ------------------- GET OUTPUT DATA --------------------------- */ 
	np = cont->rows;		/* total number of surviving points */

	plhs[0] = mxCreateNumericMatrix(np, nx, mxGetClassID(prhs[1]), mxREAL);
	ptr_d = mxGetPr(plhs[0]);
	//cvGetData( cont, (void *)ptr_d, cont->cols*8 );
	ptr_d = cont->data.db;

	cvReleaseMatHeader( &map_matrix );
}

/* --------------------------------------------------------------------------- */
void Jedge(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[], const char *method) {
	int nx, ny, nBands, i, nBytes, img_depth, kernel = 3, xord = 1, yord = 0;
	unsigned char *ptr_gray;
	double thresh1 = 40, thresh2 = 200, *ptr_d;

	IplImage *src_img = 0, *dst_img, *src_gray, *dst_16;
	mxArray *ptr_in, *ptr_out, *ptr_16;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for input and errors in user's call to function. ----------------- */
	if (!strcmp(method,"canny")) {
		if (n_in == 1) { cannyUsage();	return; }
		if (!(mxIsUint8(prhs[1]) || mxIsLogical(prhs[1])))
			mexErrMsgTxt("CANNY requires that input image is of uint8 or logical type!");
		if (n_in == 5) {
			ptr_d = mxGetPr(prhs[2]);	thresh1 = (double)ptr_d[0];
			ptr_d = mxGetPr(prhs[3]);	thresh2 = (double)ptr_d[0];
			ptr_d = mxGetPr(prhs[4]);	kernel = (int)ptr_d[0];
		}
	}
	else if (!strcmp(method,"sobel")) {
		if (n_in == 1) { sobelUsage();	return; }
		if ( !( mxIsUint8(prhs[1]) || mxIsSingle(prhs[1]) ) )
			mexErrMsgTxt("SOBEL requires image of uint8 or single type!");
		if (n_in == 5) {
			ptr_d = mxGetPr(prhs[2]);	xord = (int)ptr_d[0];
			ptr_d = mxGetPr(prhs[3]);	yord = (int)ptr_d[0];
			ptr_d = mxGetPr(prhs[4]);	kernel = (int)ptr_d[0];
		}
	}
	else {
		if (n_in == 1) { laplaceUsage();	return; }
		if ( !( mxIsUint8(prhs[1]) || mxIsSingle(prhs[1]) ) )
			mexErrMsgTxt("LAPLACE requires image of uint8 or single type!");
		if (n_in == 3) { 
			ptr_d = mxGetPr(prhs[2]);	kernel = (int)ptr_d[0];
		}
	}
	if (!(kernel == -1 || kernel == 1 || kernel == 3 || kernel == 5 || kernel == 7) )
		mexErrMsgTxt("CANNY/SOBEL/LAPLACE: Wrong kernel size!");

	if (n_out != 1) {
		mexPrintf("%s returns one (and one only) output\n", method);
		mexErrMsgTxt("");
	}
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointers for output and temporary arrays ------------------------ */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	ptr_out = mxCreateNumericMatrix(ny, nx, mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */

	src_img = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, src_img, 1, nx * nBands * nBytes );

	if (nBands == 3) {			/* Convert to GRAY */
		src_gray = cvCreateImageHeader( cvSize(nx, ny), 8, 1 );
		ptr_gray = (unsigned char *)mxCalloc (nx*ny, sizeof (unsigned char));
		cvSetImageData( src_gray, (void *)ptr_gray, nx );
		cvCvtColor(src_img, src_gray, CV_BGR2GRAY);
		for (i = 0; i < nx*ny; i++)	/* Copy the transformed image into Ctrl field */
			Ctrl->UInt8.tmp_img_in[i] = ptr_gray[i]; 
		mxFree(ptr_gray);
		cvReleaseImageHeader( &src_gray );

		/* Here we're going to cheat the src_img in order to pretend that its 2D */
		src_img->nChannels = 1;
		src_img->widthStep = nx * nBytes;
		src_img->imageSize = ny * src_img->widthStep;
	}

	dst_img = cvCreateImageHeader( cvSize(nx, ny), img_depth, 1 );
	Set_pt_Ctrl_out1 ( Ctrl, ptr_out ); 
	localSetData( Ctrl, dst_img, 2, nx * nBytes );

	if (!strcmp(method,"canny"))
		cvCanny(src_img,dst_img, thresh1, thresh2, kernel);
	else if (Ctrl->UInt8.active) {
 		Ctrl->UInt8.active = FALSE;
 		Ctrl->Int16.active = TRUE;
		ptr_16 = mxCreateNumericMatrix(ny, nx, mxINT16_CLASS, mxREAL);
		dst_16 = cvCreateImageHeader( cvSize(nx, ny), IPL_DEPTH_16S, 1 );
		Set_pt_Ctrl_out1 ( Ctrl, ptr_16 ); 
		localSetData( Ctrl, dst_16, 2, nx * 2 );
		if (!strcmp(method,"laplace"))
			cvLaplace(src_img,dst_16, kernel);
		else
			cvSobel(src_img,dst_16, xord, yord, kernel);
		// Now we have to convert the dst_16 back to UInt8 
		cvConvertScaleAbs(dst_16, dst_img, 1, 0); 
		cvReleaseImageHeader( &dst_16 );
		mxDestroyArray(ptr_16);
 		Ctrl->UInt8.active = TRUE;
 		Ctrl->Int16.active = FALSE;
	}
	else if (!strcmp(method,"laplace"))	/* When src_img is not of uint8 type */
		cvLaplace(src_img,dst_img, kernel);
	else if (!strcmp(method,"sobel"))
		cvSobel(src_img,dst_img, xord, yord, kernel);


	if (Ctrl->UInt8.active == TRUE ) {
		for (i = 0; i < nx*ny; i++)
			Ctrl->UInt8.tmp_img_out[i] = Ctrl->UInt8.tmp_img_out[i] & 1; 
	}

	cvReleaseImageHeader( &src_img );
	mxDestroyArray(ptr_in);

	if (Ctrl->UInt8.active == TRUE )
		plhs[0] = mxCreateNumericMatrix(ny, nx, mxLOGICAL_CLASS, mxREAL);
	else
		plhs[0] = mxCreateNumericMatrix(ny, nx, mxGetClassID(prhs[1]), mxREAL);

	Set_pt_Ctrl_out2 ( Ctrl, plhs[0], 1 ); 		/* Set pointer & desinterleave */

	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
	cvReleaseImageHeader( &dst_img );
	mxDestroyArray(ptr_out);
}

/* --------------------------------------------------------------------------- */
void JmorphologyEx(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	int nx, ny, nBands, nBytes, img_depth, cv_code, iterations = 1;
	const char *operation;
	double *ptr_d;

	IplImage *src_img = 0, *dst_img, *tmp_img;
	mxArray *ptr_in, *ptr_out;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for input and errors in user's call to function. ----------------- */
	if (n_in == 1) { morphologyexUsage();	return; }

	if ( !( mxIsUint8(prhs[1]) || mxIsSingle(prhs[1]) || mxIsLogical(prhs[1]) ) )
		mexErrMsgTxt("MORPHOLOGYEX requires image of uint8 or single type!");

	if (n_in < 3)
		mexErrMsgTxt("MORPHOLOGYEX requires at least 2 input arguments!");

	if(!mxIsChar(prhs[2]))
		mexErrMsgTxt("MORPHOLOGYEX: Second argument must be a valid string!");
	else
		operation = (char *)mxArrayToString(prhs[2]);

	if (!strcmp(operation,"open"))
		cv_code = CV_MOP_OPEN;
	else if (!strcmp(operation,"close"))
		cv_code = CV_MOP_CLOSE;
	else if (!strcmp(operation,"gradient"))
		cv_code = CV_MOP_GRADIENT;
	else if (!strcmp(operation,"tophat"))
		cv_code = CV_MOP_TOPHAT;
	else if (!strcmp(operation,"blackhat"))
		cv_code = CV_MOP_BLACKHAT;
	else
		mexErrMsgTxt("MORPHOLOGYEX: Unknow operation!");

	if (n_out != 1)
		mexErrMsgTxt("MORPHOLOGYEX: returns one (and one only) output");

	if (n_in == 4) {
		ptr_d = mxGetPr(prhs[3]);	iterations = (int)ptr_d[0];
	}
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointers for output and temporary arrays ------------------------ */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	ptr_out = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */
	src_img = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, src_img, 1, nx * nBands * nBytes );

	Set_pt_Ctrl_out1 ( Ctrl, ptr_out ); 
	dst_img = cvCreateImage(cvSize(nx, ny), img_depth , nBands );
	localSetData( Ctrl, dst_img, 2, nx * nBands * nBytes );

	if (cv_code == CV_MOP_GRADIENT || (iterations > 1 && (cv_code == CV_MOP_TOPHAT || cv_code == CV_MOP_BLACKHAT) )) {
		tmp_img = cvCreateImage( cvSize(nx, ny), img_depth, nBands );
	}
	else
		tmp_img = NULL;

	cvMorphologyEx(src_img,dst_img,tmp_img,NULL,cv_code,iterations);

	if (cv_code == CV_MOP_GRADIENT || (iterations > 1 && (cv_code == CV_MOP_TOPHAT || cv_code == CV_MOP_BLACKHAT) ))
		cvReleaseImage( &tmp_img );

	cvReleaseImageHeader( &src_img );
	mxDestroyArray(ptr_in);

	plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	Set_pt_Ctrl_out2 ( Ctrl, plhs[0], 1 ); 		/* Set pointer & desinterleave */

	cvReleaseImageHeader( &dst_img );
	mxDestroyArray(ptr_out);
	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}

/* --------------------------------------------------------------------------- */
void JerodeDilate(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[], const char *method) {
	int nx, ny, nBands, nBytes, img_depth, kernel = 3, iterations = 1;
	double *ptr_d;

	IplImage *src_img = 0, *dst_img;
	mxArray *ptr_in;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for input and errors in user's call to function. ----------------- */
	if (!strcmp(method,"erode"))
		if (n_in == 1) { erodeUsage();	return; }
	else if (!strcmp(method,"dilate"))
		if (n_in == 1) { dilateUsage();	return; }

	if ( !( mxIsUint8(prhs[1]) || mxIsSingle(prhs[1]) || mxIsLogical(prhs[1])) ) {
		mexPrintf("%s requires image of uint8 or single type!\n", method);
		mexErrMsgTxt("");
	}

	if (n_out != 1)
		mexErrMsgTxt("ERODE/DILATE: returns one (and one only) output");

	if (n_in == 3) {
		ptr_d = mxGetPr(prhs[2]);	iterations = (int)ptr_d[0];
	}
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointers for output and temporary arrays ------------------------ */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	//ptr_out = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  //mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */
	src_img = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, src_img, 1, nx * nBands * nBytes );

	//Set_pt_Ctrl_out1 ( Ctrl, ptr_out ); 
	Set_pt_Ctrl_out1 ( Ctrl, ptr_in ); 		/* Reuse memory */
	dst_img = cvCreateImage(cvSize(nx, ny), img_depth , nBands );
	localSetData( Ctrl, dst_img, 2, nx * nBands * nBytes );

	if (!strcmp(method,"erode"))
		cvErode(src_img,dst_img,NULL,iterations);
	else
		cvDilate(src_img,dst_img,NULL,iterations);

	//cvReleaseImageHeader( &src_img );
	//mxDestroyArray(ptr_in);

	plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	Set_pt_Ctrl_out2 ( Ctrl, plhs[0], 1 ); 		/* Set pointer & desinterleave */

	cvReleaseImageHeader( &src_img );
	cvReleaseImageHeader( &dst_img );
	mxDestroyArray(ptr_in);
	//mxDestroyArray(ptr_out);
	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}

/* --------------------------------------------------------------------------- */
void Jcolor(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	int nx, ny, nBands, nBands_out, nBytes, img_depth, cv_code, out_dims[3];
	char	*argv;

	IplImage *src_img = 0, *dst_img;
	mxArray *ptr_in, *ptr_out;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for input and errors in user's call to function. ----------------- */
	if (n_in == 1) { colorUsage();	return; }
	if (n_out > 1)
		mexErrMsgTxt("COLOR returns only one output argument!");
	if (n_in != 3)
		mexErrMsgTxt("COLOR requires 2 input arguments!");

	if(!mxIsChar(prhs[2]))
		mexErrMsgTxt("COLOR Second argument must be a valid string!");
	else {
		argv = (char *)mxArrayToString(prhs[2]);
		//<X>/<Y> = RGB, BGR, GRAY, HSV, YCrCb, XYZ, Lab, Luv, HLS
		if (!strcmp(argv,"rgb2lab"))
			cv_code = CV_BGR2Lab;
		else if (!strcmp(argv,"lab2rgb"))
			cv_code = CV_Lab2BGR;

		else if (!strcmp(argv,"rgb2luv"))
			cv_code = CV_BGR2Luv;
		else if (!strcmp(argv,"luv2rgb"))
			cv_code = CV_Luv2BGR;

		else if (!strcmp(argv,"rgb2xyz"))
			cv_code = CV_BGR2XYZ;
		else if (!strcmp(argv,"xyz2rgb"))
			cv_code = CV_XYZ2BGR;

		else if (!strcmp(argv,"rgb2yiq") || !strcmp(argv,"rgb2gray"))
			cv_code = CV_BGR2GRAY;

		else if (!strcmp(argv,"rgb2hsv"))
			cv_code = CV_BGR2HSV;
		else if (!strcmp(argv,"hsv2rgb"))
			cv_code = CV_HSV2BGR;

		else if (!strcmp(argv,"rgb2hls"))
			cv_code = CV_BGR2HLS;
		else if (!strcmp(argv,"hls2rgb"))
			cv_code = CV_HLS2BGR;

		else if (!strcmp(argv,"rgb2YCrCb"))
			cv_code = CV_BGR2YCrCb;
		else if (!strcmp(argv,"YCrCb2rgb"))
			cv_code = CV_YCrCb2BGR;

		else {
			mexPrintf("CVCOLOR ERROR: Unknown conversion type string.\n");
			mexPrintf("Valid types: rgb2lab,lab2rgb, rgb2luv,luv2rgb, rgb2xyz,xyz2rgb\n");
			mexPrintf("             rgb2yiq,yiq2rgb, rgb2hsv,luv2hsv, rgb2gray,gray2rgb\n");
			mexErrMsgTxt("             rgb2hsl,hsl2rgb, rgb2YCrCb,YCrCb2rgb.\n");
		}
	}

	if ( !(mxIsUint8(prhs[1]) || mxIsUint16(prhs[1]) || mxIsSingle(prhs[1])) )
		mexErrMsgTxt("COLOR ERROR: Invalid type. Valid types are: uint8, uint16 or single.\n");
	if (n_out != 1)
		mexErrMsgTxt("COLOR: returns one (and one only) output");
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	if (nBands == 1)
		mexErrMsgTxt("COLOR ERROR: input must be a MxNx3 array.\n");

	out_dims[0] = ny;	out_dims[1] = nx;
	if (cv_code == CV_RGB2GRAY || cv_code == CV_BGR2GRAY) {	/* rgb2gray */
		nBands_out = 1;
		out_dims[2] = nBands_out;
	}
	else {
		nBands_out = nBands;
		out_dims[2] = nBands_out;
	}
	
	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointers for output and temporary arrays ------------------------ */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	ptr_out = mxCreateNumericArray((nBands_out == 1) ? 2 : 3,
		  			out_dims, mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */
	src_img = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, src_img, 1, nx * nBands * nBytes );

	Set_pt_Ctrl_out1 ( Ctrl, ptr_out ); 
	dst_img = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands_out );
	localSetData( Ctrl, dst_img, 2, nx * nBands_out * nBytes );

	cvCvtColor(src_img, dst_img, cv_code);

	cvReleaseImageHeader( &src_img );
	mxDestroyArray(ptr_in);

	plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]), out_dims, mxGetClassID(prhs[1]), mxREAL);

	Set_pt_Ctrl_out2 ( Ctrl, plhs[0], 1 ); 		/* Set pointer & desinterleave */

	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
	cvReleaseImageHeader( &dst_img );
	mxDestroyArray(ptr_out);
}

/* --------------------------------------------------------------------------- */
void JGetQuadrangleSubPix(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	int nx, ny, nBands, nBytes, img_depth;
	double *ptr_d;

	IplImage *src, *dst;
	CvMat *map_matrix;
	mxArray *ptr_in, *ptr_out;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for errors in user's call to function. ----------------------------- */
	ptr_d = mxGetPr(prhs[2]);
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);

	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointers for output and temporary arrays ------------------------ */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	ptr_out = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */
	src = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, src, 1, nx * nBands * nBytes );

	map_matrix = cvCreateMatHeader( 2, 3, CV_64FC1 );
	//map_matrix = cvMat( 2, 3, CV_64FC1, (void *)ptr_d );
	cvSetData( map_matrix, (void *)ptr_d, 3*8 );

	Set_pt_Ctrl_out1 ( Ctrl, ptr_out ); 
	dst = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, dst, 2, nx * nBands * nBytes );

	//cvGetQuadrangleSubPix( src,  dst, map_matrix );
	cvWarpAffine( src,  dst, map_matrix,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) );

	cvReleaseImageHeader( &src );
	mxDestroyArray(ptr_in);

	plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	Set_pt_Ctrl_out2 ( Ctrl, plhs[0], 1 ); 		/* Set pointer & desinterleave */

	cvReleaseImageHeader( &dst );
	mxDestroyArray(ptr_out);
	cvReleaseMatHeader( &map_matrix );
	//cvReleaseMat( &map_matrix );
	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}

/* --------------------------------------------------------------------------- */
void Jfilter(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	int nx, ny, nBands, nBytes, img_depth, nFiltRows, nFiltCols;
	double *ptr_d, *kernel;

	IplImage *src, *dst;
	CvMat *filter;
	mxArray *ptr_in;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for errors in user's call to function. ----------------------------- */
	if (n_in == 1) { filterUsage();	return; }
	if (n_in != 3)
		mexErrMsgTxt("FILTER: requires 2 input arguments!");
	ptr_d = mxGetPr(prhs[2]);
	if ( mxIsDouble(prhs[1]) || mxIsInt8(prhs[1]) || mxIsInt32(prhs[1]) )
		mexErrMsgTxt("FILTER: IMG type not supported (supported: uint8, int16, uint16 or single)!");
	if ( !mxIsDouble(prhs[2]) )
		mexErrMsgTxt("FILTER: second argument must be a double!");
	if (n_out != 1)
		mexErrMsgTxt("FILTER: returns one (and one only) output");
	nFiltRows = mxGetM(prhs[2]);
	nFiltCols = mxGetN(prhs[2]);
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	
	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointers for output and temporary arrays ------------------------ */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	kernel = (double *)mxCalloc(nFiltCols * nFiltRows, sizeof(double));
	interleaveDouble(ptr_d, kernel, nFiltCols, nFiltRows);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */
	src = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, src, 1, nx * nBands * nBytes );

	filter = cvCreateMatHeader( nFiltRows, nFiltCols, CV_64FC1 );
	cvSetData( filter, (void *)kernel, nFiltCols*8 );

	Set_pt_Ctrl_out1 ( Ctrl, ptr_in ); 	/* Reuse the same memory */
	dst = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, dst, 2, nx * nBands * nBytes );

	cvFilter2D( src, dst, filter, cvPoint(-1,-1) ); 

	plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
	  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	Set_pt_Ctrl_out2 ( Ctrl, plhs[0], 1 ); 		/* Set pointer & desinterleave */

	cvReleaseImageHeader( &dst );
	cvReleaseImageHeader( &src );
	cvReleaseMatHeader( &filter );
	mxFree((void *)kernel);
	mxDestroyArray(ptr_in);
	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}

/* --------------------------------------------------------------------------- */
void Jsmooth(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	int nx, ny, nBands, nBytes, img_depth, par1 = 5, par2 = 0, method = CV_GAUSSIAN;
	double par3 = 0, par4 = 0;
	const char *method_s;
	IplImage *src, *dst;
	mxArray *ptr_in;
	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for errors in user's call to function. ----------------------------- */
	if (n_in == 1) { smoothUsage();	return; }

	if (n_in > 2) {
		if(!mxIsChar(prhs[2]))
			mexErrMsgTxt("CVLIB_MEX: Third argument must contain the smoothtype METHOD string!");
		else
			method_s = (char *)mxArrayToString(prhs[2]);

		if (!strcmp(method_s,"blur"))
			method = CV_BLUR;
		else if (!strcmp(method_s,"gaussian"))
			method = CV_GAUSSIAN;
		else if (!strcmp(method_s,"median"))
			method = CV_MEDIAN;
		else if (!strcmp(method_s,"bilateral"))
			method = CV_BILATERAL;
		else
			mexErrMsgTxt("SMOOTH: Unknown METHOD!");

		if (n_in >= 4 && !mxIsEmpty(prhs[3]))
			par1 = (int)(*mxGetPr(prhs[3]));
		if (n_in >= 5 && !mxIsEmpty(prhs[4]))
			par2 = (int)(*mxGetPr(prhs[4]));
		if (n_in >= 6 && !mxIsEmpty(prhs[5]))
			par3 = *mxGetPr(prhs[5]);
		if (n_in == 7 && !mxIsEmpty(prhs[6]))
			par4 = *mxGetPr(prhs[6]);
	}

	if ( !(mxIsSingle(prhs[1]) || mxIsUint8(prhs[1]) || mxIsLogical(prhs[1])) )
		mexErrMsgTxt("SMOOTH: IMG type not supported (supported: logical, uint8 or single)!");

	if ( !(mxIsUint8(prhs[1]) || mxIsLogical(prhs[1])) && (method == CV_MEDIAN || method == CV_BILATERAL) )
		mexErrMsgTxt("SMOOTH: When smoothingtype is 'median' or 'bilateral' IMG type must be logical or uint8!");

	if (n_out != 1)
		mexErrMsgTxt("SMOOTH: returns one (and one only) output");

	if ( method == CV_BILATERAL && par1 == 5 && par2 == 0 ) 	/* Use default bilateral values */
		par2 = 50;
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	
	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointers for output and temporary arrays ------------------------ */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */
	src = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, src, 1, nx * nBands * nBytes );

	Set_pt_Ctrl_out1 ( Ctrl, ptr_in ); 	/* Reuse the same memory */
	dst = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, dst, 2, nx * nBands * nBytes );

	cvSmooth( src, dst, method, par1, par2, par3, par4 );	

	plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
	  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	Set_pt_Ctrl_out2 ( Ctrl, plhs[0], 1 ); 		/* Set pointer & desinterleave */

	cvReleaseImageHeader( &dst );
	cvReleaseImageHeader( &src );
	mxDestroyArray(ptr_in);
	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}

/* --------------------------------------------------------------------------- */
void Jegipt(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[], const char *op) {
	int nx, ny, nxOut, nyOut, nBands, nBytes, img_depth, out_dims[3];

	IplImage *src, *dst;
	mxArray *ptr_in, *ptr_out;

	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for errors in user's call to function. ----------------------------- */
	if (!strcmp(op,"pyrU"))
		if (n_in == 1) { pyrUUsage();	return; }
	else if (!strcmp(op,"pyrD"))
		if (n_in == 1) { pyrDUsage();	return; }

	if ( mxIsInt8(prhs[1]) || mxIsUint32(prhs[1]) || mxIsInt32(prhs[1]) )
		mexErrMsgTxt("EGIPT: IMG type not supported (supported: uint8, uint16, int16, single or double)!");
	if (n_out != 1)
		mexErrMsgTxt("EGIPT: returns one (and one only) output");
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	
	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointer for temporary array ------------------------------------- */
	ptr_in  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in ( Ctrl, prhs[1], ptr_in, 1 ); 	/* Set pointer & interleave */
	src = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, src, 1, nx * nBands * nBytes );

	if (!strcmp(op,"pyrU")) {
		nxOut = 2*nx;		nyOut = 2*ny;
	} 
	else {
		nxOut = (int)(nx/2);	nyOut = (int)(ny/2);
	}

	/* ------ Create pointer for output array ---------------------------------------- */
	out_dims[0] = nyOut;	out_dims[1] = nxOut;	out_dims[2] = nBands;
	ptr_out = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  			out_dims, mxGetClassID(prhs[1]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_out1 ( Ctrl, ptr_out ); 	/* Set correct type pointer */
	dst = cvCreateImageHeader(cvSize(nxOut, nyOut), img_depth, nBands );
	localSetData( Ctrl, dst, 2, nxOut * nBands * nBytes );

	if (!strcmp(op,"pyrU"))
		cvPyrUp(src, dst, CV_GAUSSIAN_5x5);
	else
		cvPyrDown(src, dst, CV_GAUSSIAN_5x5);

	cvReleaseImageHeader( &src );
	mxDestroyArray(ptr_in);

	plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
					out_dims, mxGetClassID(prhs[1]), mxREAL);

	Set_pt_Ctrl_out2 ( Ctrl, plhs[0], 1 ); 		/* Set pointer & desinterleave */

	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
	cvReleaseImageHeader( &dst );
	mxDestroyArray(ptr_out);
}

/* --------------------------------------------------------------------------- */
void Jarithm(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[], const char *op) {
	int nx, nx2, ny, ny2, nBands, nBands2, nBytes, img_depth, inplace = FALSE, error = 0;
	IplImage *src1, *src2, *dst;
	CvScalar value;
	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for errors in user's call to function. ----------------------------- */
	if (n_in == 1) { arithmUsage();	return; }
	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	ny2 = mxGetM(prhs[2]);	nx2 = getNK(prhs[2],1);	nBands2 = getNK(prhs[2],2);
	if (nx*ny == 1)
		mexErrMsgTxt("CVLIB_MEX: First argument cannot be a scalar!");
	if (ny != ny2 && ny2 != 1) error++;
	if (nx != nx2 && nx2 != 1) error++;
	if (error)
		mexErrMsgTxt("CVLIB_MEX: Matrix dimensions must agree!");

	if (nx2*ny2 == 1 && (strcmp(op,"addS") && strcmp(op,"subS")) )
		mexErrMsgTxt("CVLIB_MEX: only 'addS' or 'subS' are allowed when second arg is a scalar!");

	if (n_out == 0)
		inplace = TRUE;
	/* -------------------- End of parsing input ------------------------------------- */

	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	src1 = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	cvSetImageData( src1, (void *)mxGetData(prhs[1]), nx * nBytes * nBands );
	if (nx2*ny2 != 1) { 
		src2 = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
		cvSetImageData( src2, (void *)mxGetData(prhs[2]), nx * nBytes * nBands );
	}
	else
		value.val[0] = *(double *)mxGetData(prhs[2]);

	if (!inplace) {
		plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
		dst = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
		cvSetImageData( dst, (void *)mxGetData(plhs[0]), nx * nBytes * nBands );
		if (!strcmp(op,"add"))
			cvAdd( src1, src2, dst, NULL ); 
		else if (!strcmp(op,"addS"))
			cvAddS( src1, value, dst, NULL ); 
		else if (!strcmp(op,"sub"))
			cvSub( src1, src2, dst, NULL ); 
		else if (!strcmp(op,"subS"))
			cvSubS( src1, value, dst, NULL ); 
		else if (!strcmp(op,"mul"))
			cvMul( src1, src2, dst, 1 ); 
		else if (!strcmp(op,"div"))
			cvDiv( src1, src2, dst, 1 ); 

		cvReleaseImageHeader( &dst );
	}
	else {
		if (!strcmp(op,"add"))
			cvAdd( src1, src2, src1, NULL ); 
		else if (!strcmp(op,"addS"))
			cvAddS( src1, value, src1, NULL ); 
		else if (!strcmp(op,"sub"))
			cvSub( src1, src2, src1, NULL ); 
		else if (!strcmp(op,"subS"))
			cvSubS( src1, value, src1, NULL ); 
		else if (!strcmp(op,"mul"))
			cvMul( src1, src2, src1, 1 ); 
		else if (!strcmp(op,"div"))
			cvDiv( src1, src2, src1, 1 ); 
	}

	cvReleaseImageHeader( &src1 );
	if (nx2*ny2 != 1)
		cvReleaseImageHeader( &src2 );
	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}

/* --------------------------------------------------------------------------- */
void JaddWeighted(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	int nx, ny, nx2, ny2, nz2, nBands, nBytes, img_depth, inplace = FALSE;
	double *ptr_d, alpha, beta, gamma = 0;
	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);
	IplImage *src1, *src2, *dst;

	/* ---- Check for errors in user's call to function. ----------------------------- */
	if (n_in == 1) { addWeightedUsage();	return; }
	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	ny2 = mxGetM(prhs[3]);	nx2 = getNK(prhs[3],1);	nz2 = getNK(prhs[3],2);
	if (nx != nx2 || ny != ny2 || nBands != nz2)
		mexErrMsgTxt("ADDWEIGHTED ERROR: Matrix dimensions must agree!");
	if (n_in < 5)
		mexErrMsgTxt("ADDWEIGHTED ERROR: not enough input arguments!");
	ptr_d = (double *)mxGetData(prhs[2]);	alpha = ptr_d[0];
	ptr_d = (double *)mxGetData(prhs[4]);	beta  = ptr_d[0];
	if (n_in == 6) {
		ptr_d = (double *)mxGetData(prhs[5]);	gamma = ptr_d[0];
	}
	if (n_out == 0)
		inplace = TRUE;
	/* -------------------- End of parsing input ------------------------------------- */

	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	src1 = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	cvSetImageData( src1, (void *)mxGetData(prhs[1]), nx * nBytes * nBands );
	src2 = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	cvSetImageData( src2, (void *)mxGetData(prhs[3]), nx * nBytes * nBands );

	if (!inplace) {
		plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
		dst = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
		cvSetImageData( dst, (void *)mxGetData(plhs[0]), nx * nBytes * nBands );
		cvAddWeighted( src1, alpha, src2, beta, gamma, dst );	
		cvReleaseImageHeader( &dst );
	}
	else
		cvAddWeighted( src1, alpha, src2, beta, gamma, src1 );	

	cvReleaseImageHeader( &src1 );
	cvReleaseImageHeader( &src2 );
	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}

/* --------------------------------------------------------------------------- */
void Jinpaint(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	int nx, ny, nx2, ny2, nBands, nBytes, img_depth, inplace = FALSE;
	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);
	mxArray *ptr_in1, *ptr_in2;
	IplImage *src1, *src2, *dst;

	/* ---- Check for errors in user's call to function. ----------------------------- */
	if (n_in == 1) { inpaintUsage();	return; }

	/* Check that input image is of type UInt8 */
	if (!mxIsUint8(prhs[1]))
		mexErrMsgTxt("INPAINT ERROR: Invalid first input. Data type must be: UInt8.\n");
	if ( !(mxIsUint8(prhs[2]) || mxIsLogical(prhs[2])) )
		mexErrMsgTxt("INPAINT ERROR: Invalid second input. Data type must be Uint8 or Logical.\n");

	if (n_in < 3)
		mexErrMsgTxt("INPAINT ERROR: not enough input arguments!");
	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);
	ny2 = mxGetM(prhs[2]);	nx2 = getNK(prhs[2],1);
	if (nx != nx2 || ny != ny2)
		mexErrMsgTxt("INPAINT ERROR: Matrix dimensions must agree!");
	if (getNK(prhs[2],2) != 1)
		mexErrMsgTxt("INPAINT ERROR: Second arg must be a mask array, that is with only two dimensions!");

	if (n_out == 0)
		inplace = TRUE;
	/* -------------------- End of parsing input ------------------------------------- */

	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	/* ------ Create pointer for temporary array ------------------------------------- */
	ptr_in1  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
	ptr_in2  = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[2]),
		  			mxGetDimensions(prhs[2]), mxGetClassID(prhs[2]), mxREAL);
	/* ------------------------------------------------------------------------------- */ 

	Set_pt_Ctrl_in( Ctrl, prhs[1], ptr_in1, 1 ); 	/* Set pointer & interleave */
	src1 = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	localSetData( Ctrl, src1, 1, nx * nBands * nBytes );

	Set_pt_Ctrl_in( Ctrl, prhs[2], ptr_in2, 1 ); 	/* Set pointer & interleave */
	src2 = cvCreateImageHeader( cvSize(nx, ny), img_depth, 1 );
	localSetData( Ctrl, src2, 1, nx * nBytes );

	if (!inplace) {
		Set_pt_Ctrl_out1 ( Ctrl, ptr_in1 ); 		/* Reuse memory */
		dst = cvCreateImage(cvSize(nx, ny), img_depth , nBands );
		localSetData( Ctrl, dst, 2, nx * nBands * nBytes );
		cvInpaint( src1, src2, dst, 3, CV_INPAINT_TELEA); 
		plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
		Set_pt_Ctrl_out2 ( Ctrl, plhs[0], 1 ); 		/* Set pointer & desinterleave */
	}
	else {
		cvInpaint( src1, src2, src1, 3, CV_INPAINT_TELEA); 
	}

	cvReleaseImageHeader( &src1 );
	cvReleaseImageHeader( &src2 );
	mxDestroyArray(ptr_in1);
	mxDestroyArray(ptr_in2);
	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}

/* --------------------------------------------------------------------------- */
void Jflip(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) {
	int nx, ny, nBands, nBytes, img_depth, inplace = FALSE, flip_mode = 1;
	char	*argv;
	IplImage *src, *dst;
	struct CV_CTRL *Ctrl;
	void *New_Cv_Ctrl (), Free_Cv_Ctrl (struct CV_CTRL *C);

	/* ---- Check for errors in user's call to function. ----------------------------- */
	if (n_in == 1) { flipUsage();	return; }
	if(!mxIsChar(prhs[2]))
		mexErrMsgTxt("FLIP Second argument must be a valid string!");
	else {
		argv = (char *)mxArrayToString(prhs[2]);
		if (!strcmp(argv,"ud") || !strcmp(argv,"UD"))
			flip_mode = 1;
		else if (!strcmp(argv,"lr") || !strcmp(argv,"LR")) 
			flip_mode = 0;
		else if (!strcmp(argv,"both")) 
			flip_mode = -1;
		else
			mexErrMsgTxt("FLIP ERROR: unknown flipping type string.\n");
	}
	if (n_out == 0)
		inplace = TRUE;
	/* -------------------- End of parsing input ------------------------------------- */

	ny = mxGetM(prhs[1]);	nx = getNK(prhs[1],1);	nBands = getNK(prhs[1],2);

	/* Allocate and initialize defaults in a new control structure */
	Ctrl = (struct CV_CTRL *) New_Cv_Ctrl ();
	getDataType(Ctrl, prhs, &nBytes, &img_depth);

	src = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
	cvSetImageData( src, (void *)mxGetData(prhs[1]), nx * nBytes * nBands );

	if (!inplace) {
		plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]),
		  			mxGetDimensions(prhs[1]), mxGetClassID(prhs[1]), mxREAL);
		dst = cvCreateImageHeader( cvSize(nx, ny), img_depth, nBands );
		cvSetImageData( dst, (void *)mxGetData(plhs[0]), nx * nBytes * nBands );
		cvFlip( src, dst, flip_mode ); 
		cvReleaseImageHeader( &src );
	}
	else {
		cvFlip( src, NULL, flip_mode ); 
	}
	Free_Cv_Ctrl (Ctrl);	/* Deallocate control structure */
}
	
/* -------------------------------------------------------------------------------------------- */
void localSetData(struct CV_CTRL *Ctrl, IplImage* img, int dir, int step) {
	if (Ctrl->UInt8.active == TRUE)
		if (dir == 1)
			cvSetImageData( img, (void *)Ctrl->UInt8.tmp_img_in, step );
		else
			cvSetImageData( img, (void *)Ctrl->UInt8.tmp_img_out, step );
	else if (Ctrl->Int8.active == TRUE)
		if (dir == 1)
			cvSetImageData( img, (void *)Ctrl->Int8.tmp_img_in, step );
		else
			cvSetImageData( img, (void *)Ctrl->Int8.tmp_img_out, step );
	else if (Ctrl->UInt16.active == TRUE)
		if (dir == 1)
			cvSetImageData( img, (void *)Ctrl->UInt16.tmp_img_in, step );
		else
			cvSetImageData( img, (void *)Ctrl->UInt16.tmp_img_out, step );
	else if (Ctrl->Int16.active == TRUE)
		if (dir == 1)
			cvSetImageData( img, (void *)Ctrl->Int16.tmp_img_in, step );
		else
			cvSetImageData( img, (void *)Ctrl->Int16.tmp_img_out, step );
	else if (Ctrl->Int32.active == TRUE)
		if (dir == 1)
			cvSetImageData( img, (void *)Ctrl->Int32.tmp_img_in, step );
		else
			cvSetImageData( img, (void *)Ctrl->Int32.tmp_img_out, step );
	else if (Ctrl->Float.active == TRUE)
		if (dir == 1)
			cvSetImageData( img, (void *)Ctrl->Float.tmp_img_in, step );
		else
			cvSetImageData( img, (void *)Ctrl->Float.tmp_img_out, step );
	else if (Ctrl->Double.active == TRUE)
		if (dir == 1)
			cvSetImageData( img, (void *)Ctrl->Double.tmp_img_in, step );
		else
			cvSetImageData( img, (void *)Ctrl->Double.tmp_img_out, step );
}

/* -------------------------------------------------------------------------------------------- */
void interleave(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir) {
	if (Ctrl->UInt8.active == TRUE)
		interleaveUI8(Ctrl, nx, ny, nBands, dir);
	else if (Ctrl->Int8.active == TRUE)
		interleaveI8(Ctrl, nx, ny, nBands, dir);
	else if (Ctrl->UInt16.active == TRUE)
		interleaveUI16(Ctrl, nx, ny, nBands, dir);
	else if (Ctrl->Int16.active == TRUE)
		interleaveI16(Ctrl, nx, ny, nBands, dir);
	else if (Ctrl->Int32.active == TRUE)
		interleaveI32(Ctrl, nx, ny, nBands, dir);
	else if (Ctrl->Float.active == TRUE)
		interleaveF32(Ctrl, nx, ny, nBands, dir);
	else if (Ctrl->Double.active == TRUE)
		interleaveF64(Ctrl, nx, ny, nBands, dir);
}

/* -------------------------------------------------------------------------------------------- */
void interleaveDouble(double in[], double out[], int nx, int ny) {
	/* Version to be used with 2D double vars that are not in the control struct */
	int m, n, c = 0;

	for (m = 0; m < ny; m++) 
		for (n = 0; n < nx; n++)
			out[c++] = in[m + n*ny];
}

/* -------------------------------------------------------------------------------------------- */
void interleaveBlind(unsigned char in[], unsigned char out[], int nx, int ny, int nBands, int dir) {
	/* This a version to be used with vars that are not in the control struct */
	int n_xy, m, n, k, i, c = 0;

	if (dir == 1) {		/* Matlab order to b0,g0,r0, g1,g1,r1, etc ... */
		n_xy = nx*ny;
		for (m = 0; m < ny; m++) 
			for (n = 0; n < nx; n++)
				for (k = 0, i = nBands-1; k < nBands; k++, i--)
					out[c++] = in[m + n*ny + i*n_xy];
	}
	else {			/* b0,g0,r0, b1,g1,r1, etc ...  to Matlab order */
		for (k = 0, i = nBands-1; k < nBands; k++, i--)
			for (n = 0; n < nx; n++)
				for (m = 0; m < ny; m++) 
					out[c++] = in[nBands * (n + m*nx) + i];
	}
}

/* -------------------------------------------------------------------------------------------- */
void interleaveUI8(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir) {
	int n_xy, m, n, k, i, c = 0;

	if (dir == 1) {		/* Matlab order to b0,g0,r0, g1,g1,r1, etc ... */
		n_xy = nx*ny;
		for (m = 0; m < ny; m++) 
			for (n = 0; n < nx; n++)
				for (k = 0, i = nBands-1; k < nBands; k++, i--)
					Ctrl->UInt8.tmp_img_in[c++] = Ctrl->UInt8.img_in[m + n*ny + i*n_xy];
	}
	else {			/* b0,g0,r0, b1,g1,r1, etc ...  to Matlab order */
		for (k = 0, i = nBands-1; k < nBands; k++, i--)
			for (n = 0; n < nx; n++)
				for (m = 0; m < ny; m++) 
					Ctrl->UInt8.img_out[c++] = Ctrl->UInt8.tmp_img_out[nBands * (n + m*nx) + i];
	}
}

/* -------------------------------------------------------------------------------------------- */
void interleaveI8(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir) {
	int n_xy, m, n, k, i, c = 0;

	if (dir == 1) {		/* Matlab order to b0,g0,r0, g1,g1,r1, etc ... */
		n_xy = nx*ny;
		for (m = 0; m < ny; m++) 
			for (n = 0; n < nx; n++)
				for (k = 0, i = nBands-1; k < nBands; k++, i--)
					Ctrl->Int8.tmp_img_in[c++] = Ctrl->Int8.img_in[m + n*ny + i*n_xy];
	}
	else {			/* b0,g0,r0, b1,g1,r1, etc ...  to Matlab order */
		for (k = 0, i = nBands-1; k < nBands; k++, i--)
			for (n = 0; n < nx; n++)
				for (m = 0; m < ny; m++) 
					Ctrl->Int8.img_out[c++] = Ctrl->Int8.tmp_img_out[nBands * (n + m*nx) + i];
	}
}

/* -------------------------------------------------------------------------------------------- */
void interleaveUI16(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir) {
	int n_xy, m, n, k, i, c = 0;

	if (dir == 1) {		/* Matlab order to b0,g0,r0, g1,g1,r1, etc ... */
		n_xy = nx*ny;
		for (m = 0; m < ny; m++) 
			for (n = 0; n < nx; n++)
				for (k = 0, i = nBands-1; k < nBands; k++, i--)
					Ctrl->UInt16.tmp_img_in[c++] = Ctrl->UInt16.img_in[(m + n*ny + i*n_xy)];
	}
	else {			/* b0,g0,r0, b1,g1,r1, etc ...  to Matlab order */
		for (k = 0, i = nBands-1; k < nBands; k++, i--)
			for (n = 0; n < nx; n++)
				for (m = 0; m < ny; m++) 
					Ctrl->UInt16.img_out[c++] = Ctrl->UInt16.tmp_img_out[nBands * (n + m*nx) + i];
	}
}

/* -------------------------------------------------------------------------------------------- */
void interleaveI16(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir) {
	int n_xy, m, n, k, i, c = 0;

	if (dir == 1) {		/* Matlab order to b0,g0,r0, g1,g1,r1, etc ... */
		n_xy = nx*ny;
		for (m = 0; m < ny; m++) 
			for (n = 0; n < nx; n++)
				for (k = 0, i = nBands-1; k < nBands; k++, i--)
					Ctrl->Int16.tmp_img_in[c++] = Ctrl->Int16.img_in[(m + n*ny + i*n_xy)];
	}
	else {			/* b0,g0,r0, b1,g1,r1, etc ...  to Matlab order */
		for (k = 0, i = nBands-1; k < nBands; k++, i--)
			for (n = 0; n < nx; n++)
				for (m = 0; m < ny; m++) 
					Ctrl->Int16.img_out[c++] = Ctrl->Int16.tmp_img_out[nBands * (n + m*nx) + i];
	}
}

/* -------------------------------------------------------------------------------------------- */
void interleaveI32(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir) {
	int n_xy, m, n, k, i, c = 0;

	if (dir == 1) {		/* Matlab order to b0,g0,r0, g1,g1,r1, etc ... */
		n_xy = nx*ny;
		for (m = 0; m < ny; m++) 
			for (n = 0; n < nx; n++)
				for (k = 0, i = nBands-1; k < nBands; k++, i--)
					Ctrl->Int32.tmp_img_in[c++] = Ctrl->Int32.img_in[(m + n*ny + i*n_xy)];
	}
	else {			/* b0,g0,r0, b1,g1,r1, etc ...  to Matlab order */
		for (k = 0, i = nBands-1; k < nBands; k++, i--)
			for (n = 0; n < nx; n++)
				for (m = 0; m < ny; m++) 
					Ctrl->Int32.img_out[c++] = Ctrl->Int32.tmp_img_out[nBands * (n + m*nx) + i];
	}
}

/* -------------------------------------------------------------------------------------------- */
void interleaveF32(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir) {
	int n_xy, m, n, k, i, c = 0;

	if (dir == 1) {		/* Matlab order to b0,g0,r0, g1,g1,r1, etc ... */
		n_xy = nx*ny;
		for (m = 0; m < ny; m++) 
			for (n = 0; n < nx; n++)
				for (k = 0, i = nBands-1; k < nBands; k++, i--)
					Ctrl->Float.tmp_img_in[c++] = Ctrl->Float.img_in[(m + n*ny + i*n_xy)];
	}
	else {			/* b0,g0,r0, b1,g1,r1, etc ...  to Matlab order */
		for (k = 0, i = nBands-1; k < nBands; k++, i--)
			for (n = 0; n < nx; n++)
				for (m = 0; m < ny; m++) 
					Ctrl->Float.img_out[c++] = Ctrl->Float.tmp_img_out[nBands * (n + m*nx) + i];
	}
}

/* -------------------------------------------------------------------------------------------- */
void interleaveF64(struct CV_CTRL *Ctrl, int nx, int ny, int nBands, int dir) {
	int n_xy, m, n, k, i, c = 0;

	if (dir == 1) {		/* Matlab order to b0,g0,r0, g1,g1,r1, etc ... */
		n_xy = nx*ny;
		for (m = 0; m < ny; m++) 
			for (n = 0; n < nx; n++)
				for (k = 0, i = nBands-1; k < nBands; k++, i--)
					Ctrl->Double.tmp_img_in[c++] = Ctrl->Double.img_in[(m + n*ny + i*n_xy)];
	}
	else {			/* b0,g0,r0, b1,g1,r1, etc ...  to Matlab order */
		for (k = 0, i = nBands-1; k < nBands; k++, i--)
			for (n = 0; n < nx; n++)
				for (m = 0; m < ny; m++) 
					Ctrl->Double.img_out[c++] = Ctrl->Double.tmp_img_out[nBands * (n + m*nx) + i];
	}
}

/* ---------------------------------------------------------------------- */
void *New_Cv_Ctrl () {	/* Allocate and initialize a new control structure */
	struct CV_CTRL *C;
	
	C = (struct CV_CTRL *) mxCalloc (1, sizeof (struct CV_CTRL));
	return ((void *)C);
}

/* ---------------------------------------------------------------------- */
void Free_Cv_Ctrl (struct CV_CTRL *C) {	/* Deallocate control structure */
	mxFree ((void *)C);	
}

/* ---------------------------------------------------------------------- */
void Set_pt_Ctrl_in (struct CV_CTRL *Ctrl, const mxArray *pi, mxArray *pit, int interl) {
	/* Set input image pointers to correct type and, optionaly, do the interleaving */ 
	if (Ctrl->UInt8.active == TRUE) {
		Ctrl->UInt8.img_in      = (unsigned char *)mxGetData(pi);
		Ctrl->UInt8.tmp_img_in  = (unsigned char *)mxGetData(pit);
	}
	else if (Ctrl->Int8.active == TRUE) {
		Ctrl->Int8.img_in      = (char *)mxGetData(pi);
		Ctrl->Int8.tmp_img_in  = (char *)mxGetData(pit);
	}
	else if (Ctrl->UInt16.active == TRUE) {
		Ctrl->UInt16.img_in      = (unsigned short int *)mxGetData(pi);
		Ctrl->UInt16.tmp_img_in  = (unsigned short int *)mxGetData(pit);
	}
	else if (Ctrl->Int16.active == TRUE) {
		Ctrl->Int16.img_in      = (short int *)mxGetData(pi);
		Ctrl->Int16.tmp_img_in  = (short int *)mxGetData(pit);
	}
	else if (Ctrl->Int32.active == TRUE) {
		Ctrl->Int32.img_in      = (int *)mxGetData(pi);
		Ctrl->Int32.tmp_img_in  = (int *)mxGetData(pit);
	}
	else if (Ctrl->Float.active == TRUE) {
		Ctrl->Float.img_in      = (float *)mxGetData(pi);
		Ctrl->Float.tmp_img_in  = (float *)mxGetData(pit);
	}
	else if (Ctrl->Double.active == TRUE) {
		Ctrl->Double.img_in      = (double *)mxGetData(pi);
		Ctrl->Double.tmp_img_in  = (double *)mxGetData(pit);
	}

	if (interl)
		interleave (Ctrl, getNK(pi,1), mxGetM(pi), getNK(pi,2), 1);
}

/* ---------------------------------------------------------------------- */
void Set_pt_Ctrl_out1 ( struct CV_CTRL *Ctrl, mxArray *pi ) {
	/* Set output tmp image pointer to correct type */ 
	if (Ctrl->UInt8.active == TRUE) {
		Ctrl->UInt8.tmp_img_out  = (unsigned char *)mxGetData(pi);
	}
	else if (Ctrl->Int8.active == TRUE) {
		Ctrl->Int8.tmp_img_out   = (char *)mxGetData(pi);
	}
	else if (Ctrl->UInt16.active == TRUE) {
		Ctrl->UInt16.tmp_img_out = (unsigned short int *)mxGetData(pi);
	}
	else if (Ctrl->Int16.active == TRUE) {
		Ctrl->Int16.tmp_img_out  = (short int *)mxGetData(pi);
	}
	else if (Ctrl->Int32.active == TRUE) {
		Ctrl->Int32.tmp_img_out  = (int *)mxGetData(pi);
	}
	else if (Ctrl->Float.active == TRUE) {
		Ctrl->Float.tmp_img_out  = (float *)mxGetData(pi);
	}
	else if (Ctrl->Double.active == TRUE) {
		Ctrl->Double.tmp_img_out = (double *)mxGetData(pi);
	}
}

/* ---------------------------------------------------------------------- */
void Set_pt_Ctrl_out2 (struct CV_CTRL *Ctrl, mxArray *po, int interl) {
	/* Set output image pointers to correct type and, optionaly, do the interleaving */ 
	if (Ctrl->UInt8.active == TRUE) {
		Ctrl->UInt8.img_out  = (unsigned char *)mxGetData(po);
	}
	else if (Ctrl->Int8.active == TRUE) {
		Ctrl->Int8.img_out   = (char *)mxGetData(po);
	}
	else if (Ctrl->UInt16.active == TRUE) {
		Ctrl->UInt16.img_out = (unsigned short int *)mxGetData(po);
	}
	else if (Ctrl->Int16.active == TRUE) {
		Ctrl->Int16.img_out  = (short int *)mxGetData(po);
	}
	else if (Ctrl->Int32.active == TRUE) {
		Ctrl->Int32.img_out  = (int *)mxGetData(po);
	}
	else if (Ctrl->Float.active == TRUE) {
		Ctrl->Float.img_out  = (float *)mxGetData(po);
	}
	else if (Ctrl->Double.active == TRUE) {
		Ctrl->Double.img_out = (double *)mxGetData(po);
	}

	if (interl)
		interleave (Ctrl, getNK(po,1), mxGetM(po), getNK(po,2), -1);
}

/* ---------------------------------------------------------------------- */
int getNK(const mxArray *p, int which) {
	/* Get number of columns or number of bands of a mxArray */
	int nx, nBands, nDims;
	const int *dim_array;

	nDims     = mxGetNumberOfDimensions(p);
	dim_array = mxGetDimensions(p);
	nx = dim_array[1];
	nBands = dim_array[2];
	if (nDims == 2) 	/* Otherwise it would stay undefined */
		nBands = 1;

	if (which == 1)
		return(nx);
	else if (which == 2)
		return(nBands);
	else
		mexErrMsgTxt("getNK: Bad dimension number!");
}

/* ---------------------------------------------------------------------- */
void getDataType(struct CV_CTRL *Ctrl, const mxArray *prhs[], int *nBytes, int *img_depth) {
	/* Find out in which data type was given the input array */
	int error = 0;

	if (mxIsComplex(prhs[1]) || mxIsSparse(prhs[1]))
		mexErrMsgTxt("Input image must be a real matrix");

	if (mxIsLogical(prhs[1])) {		/* Logicals take precedence over UInt8 */
		*img_depth = IPL_DEPTH_8U;
		*nBytes    = 1;
		Ctrl->UInt8.active = TRUE;
	}
	else if (mxIsUint8(prhs[1])) {
		*img_depth = IPL_DEPTH_8U;
		*nBytes    = 1;
		Ctrl->UInt8.active = TRUE;
	}
	else if (mxIsInt8(prhs[1])) {
		*img_depth = IPL_DEPTH_8S;
		*nBytes    = 1;
		Ctrl->Int8.active = TRUE;
		error++;
	}
	else if (mxIsUint16(prhs[1])) {
		*img_depth = IPL_DEPTH_16U;
		*nBytes    = 2;
		Ctrl->UInt16.active = TRUE;
	}
	else if (mxIsInt16(prhs[1])) {
		*img_depth = IPL_DEPTH_16S;
		*nBytes    = 2;
		Ctrl->Int16.active = TRUE;
		error++;
	}
	else if (mxIsInt32(prhs[1])) {
		*img_depth = IPL_DEPTH_32S;
		*nBytes    = 4;
		Ctrl->Int32.active = TRUE;
		error++;
	}
	else if (mxIsSingle(prhs[1])) {
		*img_depth = IPL_DEPTH_32F;
		*nBytes    = 4;
		Ctrl->Float.active = TRUE;
	}
	else if (mxIsDouble(prhs[1])) {
		*img_depth = IPL_DEPTH_64F;
		*nBytes    = 8;
		Ctrl->Double.active = TRUE;
		error++;
	}
	else {
		mexPrintf("CVLIB_MEX ERROR: Invalid input data type.\n");
		mexErrMsgTxt("Valid types are: double, single, Int32, UInt16, Int16, and Uint8.\n");
	}
	//mexPrintf("Merda %d %d %d %d\n",(int)IPL_DEPTH_8S, IPL_DEPTH_16S, IPL_DEPTH_32S, IPL_DEPTH_SIGN);

	/* For a very obscure reason the types Int8,Int16,Int32,Double are reported illegal
	   in cxarray (line 3346), but the most strange is that they are foreseen there */
	/*if (error) {
		mexPrintf("CVRESIZE_MEX ERROR: Invalid input data type.\n");
		mexErrMsgTxt("Valid types are: double, single, UInt16, Int16, and Uint8.\n");
	}*/
}

/* ---------------------------------------------------------------------- */
void cvResizeUsage() {
	mexPrintf("Usage: B = cvlib_mex('resize',A,M,METHOD);\n");
	mexPrintf("       returns an image that is M times the size of A.\n");
	mexPrintf("       If M is between 0 and 1.0, B is smaller than A. If\n");
	mexPrintf("       M is greater than 1.0, B is larger than A. If METHOD is\n");
	mexPrintf("       omitted, the bilinear interpolation is used.\n\n");
 
	mexPrintf("B = cvlib_mex('resize',A,[MROWS MCOLS],METHOD) returns an image of size\n");
	mexPrintf("       MROWS-by-MCOLS. The available METHODS are:\n");
	mexPrintf("       'bilinear' (default) bilinear interpolation\n");
	mexPrintf("       'bicubic'  bicubic interpolation\n");
	mexPrintf("       'nearest'  nearest neighbor interpolation\n");
	mexPrintf("       'area'     resampling using pixel area relation. It is preferred method\n");
	mexPrintf("             for image decimation that gives moire-free results.\n\n");

	mexPrintf("       Class support: The finest example of OpenCV data type support mess:\n");
	mexPrintf("             If interpolation METHOD is 'nearest' all data types are supported:\n");
	mexPrintf("             Otherwise: logical, uint8, uint16 and single!\n");
	mexPrintf("       Memory overhead: 1 copy of IMG and 1 copy of B.\n");
}

/* -------------------------------------------------------------------------------------------- */
void floodFillUsage() {
	mexPrintf("Usage: B = cvlib_mex('floodfill',IMG,PARAMS_STRUCT);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 rgb OR a MxN intensity image:\n");
	mexPrintf("       PARAMS_STRUCT is a structure with the following fields (only Point is mandatory):\n");
	mexPrintf("       Point -> a 1x2 vector with the X,Y coordinates (in the pixel\n");
	mexPrintf("                reference frame) of the selected image point\n");
	mexPrintf("       Tolerance -> a scalar in the [0 255] interval [Default is 20]\n");
	mexPrintf("       Connect   -> a scalar with the connectivity, either 4 or 8 [Default is 4]\n");
	mexPrintf("       FillColor -> a 1x3 vector with the fill color [default picks a random color]\n\n");
	mexPrintf("       Alternatively you may also call cvfill_mex in this way:\n");
	mexPrintf("       B = cvfill_mex('floodfill',IMG,POINT);\n");
	mexPrintf("       where POINT is a 1x2 vector with the X,Y coordinates as described above\n\n");

	mexPrintf("       Class support: logical or uint8.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG.\n");
}

/* -------------------------------------------------------------------------------------------- */
void lineUsage() {
	mexPrintf("Usage: cvlib_mex('line',IMG,PT1,PT2,[COLOR,THICK,LINE_TYPE]);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 rgb OR a MxN intensity image:\n");
	mexPrintf("       draws, inplace, the line segment between PT1 and PT2 points in the image.\n");
	mexPrintf("       IM2 = cvlib_mex('line',IMG,PT1,PT2,[COLOR,THICK,LINE_TYPE]);\n");
	mexPrintf("       Returns the drawing in the the new array IM2.\n\n");
	mexPrintf("       Terms inside brakets are optional and can be empty,\n");
	mexPrintf("       e.g (...,[],[],LINE_TYPE) or (...,[],5) are allowed.\n");
	mexPrintf("	  PT1 & PT2 -> Start and end points of the line segment. Note PT is a 1x2 vector e.g [x y]\n");
	mexPrintf("       COLOR -> Line color. Can be a 1x3 vector, e.g. the default [255 255 255], or a scalar (gray).\n");
	mexPrintf("       THICK -> Line thickness (default 1)\n");
	mexPrintf("       LINE_TYPE -> Type of line. 8 - 8-connected line (default), 4 - 4-connected, 16 - antialiased.\n\n");

	mexPrintf("       Class support: uint8.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG.\n");
}

/* -------------------------------------------------------------------------------------------- */
void rectUsage() {
	mexPrintf("Usage: cvlib_mex('rectangle',IMG,PT1,PT2,[COLOR,THICK,LINE_TYPE]);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 rgb OR a MxN intensity image:\n");
	mexPrintf("       draws, inplace, a rectangle between the oposit corners PT1 and PT2 in the image.\n");
	mexPrintf("       IM2 = cvlib_mex('rectangle',IMG,PT1,PT2,[COLOR,THICK,LINE_TYPE]);\n");
	mexPrintf("       Returns the drawing in the the new array IM2.\n\n");
	mexPrintf("       Terms inside brakets are optional and can be empty,\n");
	mexPrintf("       e.g (...,[],[],LINE_TYPE) or (...,[],5) are allowed.\n");
	mexPrintf("	  PT1 & PT2 -> Oposit rectangle vertices. Note PT is a 1x2 vector e.g [x y]\n");
	mexPrintf("       COLOR -> Line color. Can be a 1x3 vector, e.g. the default [255 255 255], or a scalar (gray).\n");
	mexPrintf("       THICK -> Line thickness (default 1). If negative a filled rectangle is drawn.\n");
	mexPrintf("       LINE_TYPE -> Type of line. 8 - 8-connected line (default), 4 - 4-connected, 16 - antialiased.\n\n");

	mexPrintf("       Class support: uint8.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG.\n");
}
/* -------------------------------------------------------------------------------------------- */
void circUsage() {
	mexPrintf("Usage: cvlib_mex('circle',IMG,CENTER,RADIUS,[COLOR,THICK,LINE_TYPE]);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 rgb OR a MxN intensity image:\n");
	mexPrintf("       draws, inplace, a simple or filled circle with given center and radius.\n");
	mexPrintf("       IM2 = cvlib_mex('circle',IMG,CENTER,RADIUS,[COLOR,THICK,LINE_TYPE]);\n");
	mexPrintf("       Returns the drawing in the the new array IM2.\n\n");
	mexPrintf("       Terms inside brakets are optional and can be empty,\n");
	mexPrintf("       e.g (...,[],[],LINE_TYPE) or (...,[],5) are allowed.\n");
	mexPrintf("	  CENTER & RADIUS -> Note, they are both 1x2 vectors e.g [x y]\n");
	mexPrintf("       COLOR -> Line color. Can be a 1x3 vector, e.g. the default [255 255 255], or a scalar (gray).\n");
	mexPrintf("       THICK -> Line thickness (default 1). If negative a filled circle is drawn.\n");
	mexPrintf("       LINE_TYPE -> Type of line. 8 - 8-connected line (default), 4 - 4-connected, 16 - antialiased.\n\n");

	mexPrintf("       Class support: uint8.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG.\n");
}

/* -------------------------------------------------------------------------------------------- */
void eBoxUsage() {
	mexPrintf("Usage: cvlib_mex('eBox',IMG,BOX,[COLOR,THICK,LINE_TYPE]);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 rgb OR a MxN intensity image:\n");
	mexPrintf("       draws, inplace, draws a simple or thick ellipse outline, or fills an ellipse.\n");
	mexPrintf("	  BOX -> Structure with the following fields;\n");
	mexPrintf("	         'center' -> a 2 element vector with the ellipse center coords (decimal pixeis);\n");
	mexPrintf("	         'size'   -> a 2 element vector with the ellipse WIDTH and HEIGHT (decimal pixeis);\n");
	mexPrintf("	         'angle'  -> the angle between the horizontal axis and the WIDTH (degrees);\n");
	mexPrintf("	                     If this field is not present, angle defaults to zero.\n");
	mexPrintf("       IM2 = cvlib_mex('eBox',IMG,BOX,[COLOR,THICK,LINE_TYPE]);\n");
	mexPrintf("       Returns the drawing in the the new array IM2.\n\n");
	mexPrintf("       Terms inside brakets are optional and can be empty,\n");
	mexPrintf("       e.g (...,[],[],LINE_TYPE) or (...,[],5) are allowed.\n");
	mexPrintf("       COLOR -> Line color. Can be a 1x3 vector, e.g. the default [255 255 255], or a scalar (gray).\n");
	mexPrintf("       THICK -> Line thickness (default 1). If negative a filled ellipse is drawn.\n");
	mexPrintf("       LINE_TYPE -> Type of line. 8 - 8-connected line (default), 4 - 4-connected, 16 - antialiased.\n\n");

	mexPrintf("       Class support: uint8.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG.\n");
}

/* -------------------------------------------------------------------------------------------- */
void plineUsage() {
	mexPrintf("Usage: cvlib_mex('polyline',IMG,PT,[COLOR,THICK,LINE_TYPE]);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 rgb OR a MxN intensity image:\n");
	mexPrintf("       draws, inplace, the polyline whose vertex are contained in the Mx2 or 2xN PT array.\n");
	mexPrintf("       If PT is a cell vector with N elements, draws N polylines in the image.\n");
	mexPrintf("       Each cell element must contain a Mx2 OR 2xN array with the x,y pixel coords\n");
	mexPrintf("       of the polyline to be ploted.\n");
	mexPrintf("       IM2 = cvlib_mex('polyline',IMG,PT,[COLOR,THICK,LINE_TYPE]);\n");
	mexPrintf("       Returns the drawing in the the new array IM2.\n\n");
	mexPrintf("       Terms inside brakets are optional and can be empty,\n");
	mexPrintf("       e.g (...,[],[],LINE_TYPE) or (...,[],5) are allowed.\n");
	mexPrintf("       COLOR -> Line color. Can be a 1x3 vector, e.g. the default [255 255 255], or a scalar (gray).\n");
	mexPrintf("       THICK -> Line thickness (default 1)\n");
	mexPrintf("       LINE_TYPE -> Type of line. 8 - 8-connected line (default), 4 - 4-connected, 16 - antialiased.\n\n");

	mexPrintf("       Class support: uint8.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG.\n");
}

/* -------------------------------------------------------------------------------------------- */
void fillPlineUsage() {
	mexPrintf("Usage: cvlib_mex('fillpoly',IMG,PT,[COLOR,THICK,LINE_TYPE]);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 rgb OR a MxN intensity image:\n");
	mexPrintf("       fills an area bounded by several polygonal contours inplace.\n");
	mexPrintf("       The polyligonal contour vertex are contained in the Mx2 or 2xN PT array.\n");
	mexPrintf("       If PT is a cell vector with N elements, fills N polygons in the image.\n");
	mexPrintf("       Each cell element must contain a Mx2 OR 2xN array with the x,y pixel coords\n");
	mexPrintf("       of the polygon to be filled.\n");
	mexPrintf("       IM2 = cvlib_mex('polyline',IMG,PT,[COLOR,THICK,LINE_TYPE]);\n");
	mexPrintf("       Returns the drawing in the the new array IM2.\n\n");
	mexPrintf("       Terms inside brakets are optional and can be empty,\n");
	mexPrintf("       e.g (...,[],[],LINE_TYPE) or (...,[],5) are allowed.\n");
	mexPrintf("       COLOR -> Line color. Can be a 1x3 vector, e.g. the default [255 255 255], or a scalar (gray).\n");
	mexPrintf("       THICK -> Line thickness (default 1)\n");
	mexPrintf("       LINE_TYPE -> Type of line. 8 - 8-connected line (default), 4 - 4-connected, 16 - antialiased.\n\n");

	mexPrintf("       Class support: uint8.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG.\n");
}

/* -------------------------------------------------------------------------------------------- */
void fillConvUsage() {
	mexPrintf("Usage: cvlib_mex('fillconvex',IMG,PT,[COLOR,THICK,LINE_TYPE]);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 rgb OR a MxN intensity image:\n");
	mexPrintf("       fills an area bounded by a convex polygonal interior inplace.\n");
	mexPrintf("       The polyligonal contour vertex are contained in the Mx2 or 2xN PT array.\n");
	mexPrintf("       IM2 = cvlib_mex('polyline',IMG,PT,[COLOR,THICK,LINE_TYPE]);\n");
	mexPrintf("       Returns the drawing in the the new array IM2.\n\n");
	mexPrintf("       Terms inside brakets are optional and can be empty,\n");
	mexPrintf("       e.g (...,[],[],LINE_TYPE) or (...,[],5) are allowed.\n");
	mexPrintf("       COLOR -> Line color. Can be a 1x3 vector, e.g. the default [255 255 255], or a scalar (gray).\n");
	mexPrintf("       THICK -> Line thickness (default 1)\n");
	mexPrintf("       LINE_TYPE -> Type of line. 8 - 8-connected line (default), 4 - 4-connected, 16 - antialiased.\n\n");

	mexPrintf("       Class support: uint8.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG.\n");
}

/* -------------------------------------------------------------------------------------------- */
void goodFeaturesUsage() {
	mexPrintf("Usage: B = cvlib_mex('goodfeatures',IMG,[,M,QUALITY,DIST]);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 or a MxN intensity image:\n");
	mexPrintf("       returns strong corners on image.\n");
	mexPrintf("       Terms inside brakets are optional and can be empty,\n");
	mexPrintf("       e.g (...,[],[],DIST) is allowed.\n");
	mexPrintf("       M  - number of maximum output corners [default is all up to 10000]\n");
	mexPrintf("       QUALITY - only those corners are selected, which minimal eigen value is\n");
	mexPrintf("           non-less than maximum of minimal eigen values on the image,\n");
	mexPrintf("           multiplied by quality_level. For example, quality_level = 0.1\n");
	mexPrintf("           means that selected corners must be at least 1/10 as good as\n");
	mexPrintf("           the best corner. [Default is 0.1]\n");
	mexPrintf("       DIST - The selected corners(after thresholding using quality_level)\n");
	mexPrintf("           are rerified such that pair-wise distance between them is\n");
	mexPrintf("           non-less than min_distance [Default is 10]\n\n");

	mexPrintf("       Class support: uint8.\n");
	mexPrintf("       Memory overhead: 2 arrays (single) with size MxN.\n");
}

/* -------------------------------------------------------------------------------------------- */
void houghLines2Usage() {
	mexPrintf("Usage: B = cvlib_mex('houghlines2',IMG);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 or MxN intensity image OR a MxN logical (mask):\n");
	mexPrintf("       B is a P-by-1 cell array where P is the number of detected lines.\n");
	mexPrintf("       Each cell in the cell array contains a 2x2 matrix. Each row in the\n");
	mexPrintf("       matrix contains the row and column pixel coordinates of the line.\n\n");

	mexPrintf("       Class support: logical or uint8.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG and, if IMG is not of type logical, a MxN.\n");
}

/* -------------------------------------------------------------------------------------------- */
void houghCirclesUsage() {
	mexPrintf("Usage: B = cvlib_mex('houghcircles',IMG [,DP,MIN_DIST,PAR1,PAR2,R0,R1]);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 or MxN intensity image.\n");
	mexPrintf("       B is a P-by-3 matrix where P is the number of detected circles.\n");
	mexPrintf("       and each row of B contains (x,y,radius)\n");
	mexPrintf("       With the options below you may provide empties ([]) to use the default value.\n");
	mexPrintf("	  DP -> Resolution of the accumulator used to detect centers of the circles. For example\n");
	mexPrintf("	        if it is 1, the accumulator will have the same resolution as the input image,\n");
	mexPrintf("	        if it is 2 - accumulator will have twice smaller width and height, etc. [Default = 1]\n");
	mexPrintf("	  MIN_DIST -> Minimum distance between centers of the detected circles. If the parameter\n");
	mexPrintf("	        is too small, multiple neighbor circles may be falsely detected in addition\n");
	mexPrintf("	        to a true one. If it is too large, some circles may be missed [Default = 20]\n");
	mexPrintf("	  PAR1 -> higher threshold of the two passed to Canny edge detector [Default = 50]\n");
	mexPrintf("	  PAR2 -> accumulator threshold at the center detection stage. The smaller it is,\n");
	mexPrintf("	        the more false circles may be detected. Circles, corresponding to the\n");
	mexPrintf("	        larger accumulator values, will be returned first. [Default = 60]\n\n");
	mexPrintf("	  R0 -> Minimal radius of the circles to search for (pixels) [Default = 5]\n");
	mexPrintf("	  R1 -> Maximal radius of the circles to search for. By default the maximal radius\n");
	mexPrintf("	        is set to max(image_width, image_height). [Default = 0]\n");

	mexPrintf("       Class support: uint8.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG.\n");
}

/* -------------------------------------------------------------------------------------------- */
void findContoursUsage() {
	mexPrintf("Usage: B = cvlib_mex('contours',IMG);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 or MxN intensity image OR a MxN logical (mask):\n");
	mexPrintf("       B is a P-by-1 cell array where P is the number of detected lines.\n");
	mexPrintf("       Each cell in the cell array contains a Qx2 matrix. Each row in the\n");
	mexPrintf("       matrix contains the row and column pixel coordinates of the line.\n\n");

	mexPrintf("       Class support: logical or uint8.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG and, if IMG is not of type logical, a MxN.\n");
}

/* -------------------------------------------------------------------------------------------- */
void cannyUsage() {
	mexPrintf("Usage: C = cvlib_mex('canny',IMG);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 rgb OR a MxN intensity image:\n");
	mexPrintf("       C = cvlib_mex('canny',IMG,threshold1,threshold2,aperture_size);\n");
	mexPrintf("       If not provided threshold1 = 40, threshold2 = 200, aperture_size = 3;\n\n");

	mexPrintf("       Class support: uint8.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG and a MxN uint8 matrix.\n");
}

/* -------------------------------------------------------------------------------------------- */
void sobelUsage() {
	mexPrintf("Usage: C = cvlib_mex('sobel',IMG);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 rgb OR a MxN intensity image OR a MxN single:\n");
	mexPrintf("       C = cvlib_mex('sobel',IMG,xorder,yorder,aperture_size);\n");
	mexPrintf("       If not provided xorder = 1, yorder = 0, aperture_size = 3;\n\n");

	mexPrintf("       Class support: uint8 or single.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG and a MxN matrix of the same type as IMG.\n");
}

/* -------------------------------------------------------------------------------------------- */
void laplaceUsage() {
	mexPrintf("Usage: C = cvlib_mex('laplace',IMG);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 rgb OR a MxN intensity image OR a MxN single:\n");
	mexPrintf("       C = cvlib_mex('laplace',IMG,aperture_size);\n");
	mexPrintf("       If not provided aperture_size = 3;\n\n");

	mexPrintf("       Class support: uint8 or single.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG and a MxN matrix of the same type as IMG.\n");
}

/* -------------------------------------------------------------------------------------------- */
void erodeUsage() {
	mexPrintf("Usage: C = cvlib_mex('erode',IMG);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 rgb OR a MxN intensity image OR a MxN single:\n");
	mexPrintf("       C = cvlib_mex('erode',IMG,iterations);\n");
	mexPrintf("       If not provided iterations = 1;\n\n");

	mexPrintf("       Class support: logical, uint8 or single.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG.\n");
}

/* -------------------------------------------------------------------------------------------- */
void dilateUsage() {
	mexPrintf("Usage: C = cvlib_mex('dilate',IMG);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 rgb OR a MxN intensity image OR a MxN single:\n");
	mexPrintf("       C = cvlib_mex('dilate',IMG,iterations);\n");
	mexPrintf("       If not provided iterations = 1;\n\n");

	mexPrintf("       Class support: logical, uint8 or single.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG.\n");
}

/* -------------------------------------------------------------------------------------------- */
void morphologyexUsage() {
	mexPrintf("Usage: C = cvlib_mex('morphologyex',IMG,OPERATION);\n");
	mexPrintf("       where IMG is a uint8 MxNx3 rgb OR a MxN intensity image OR a MxN single:\n");
	mexPrintf("	  The available OPERATIONS are:\n");
	mexPrintf("	  'open'\n");
	mexPrintf("	  'close'\n");
	mexPrintf("	  'gradient'  dilate - erode\n");
	mexPrintf("	  'tophat'    IMG - open\n");
	mexPrintf("	  'blackhat'  close - IMG\n\n");
	mexPrintf("       C = cvlib_mex('morphologyex',IMG,OPERATION,iterations);\n");
	mexPrintf("       If not provided iterations = 1;\n\n");

	mexPrintf("       Class support: logical, uint8 or single.\n");
	mexPrintf("       Memory overhead: 2 copies of IMG.\n");
}

/* -------------------------------------------------------------------------------------------- */
void colorUsage() {
	mexPrintf("Usage: B = cvlib_mex('color',IMG,'TRF');\n");
	mexPrintf("       where IMG is a MxNx3 image of type: uint8, uint16 OR single (0..1 interval).\n");
	mexPrintf("       TRF is a string controling the transformation. Possibilities are:\n");
	mexPrintf("       rgb2lab,lab2rgb, rgb2luv,luv2rgb, rgb2xyz,xyz2rgb\n");
	mexPrintf("       rgb2yiq,yiq2rgb, rgb2hsv,luv2hsv, rgb2hsl,hsl2rgb, rgb2YCrCb,YCrCb2rgb\n\n");

	mexPrintf("       Class support: uint8, uint16 or single.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG and 1 copy of B.\n");
}

/* -------------------------------------------------------------------------------------------- */
void flipUsage() {
	mexPrintf("Usage: B = cvlib_mex('flip',IMG,'DIR');\n");
	mexPrintf("       Flip a 2D array around vertical, horizontall or both axis\n");
	mexPrintf("       where IMG is an array of any type and DIR = 'ud', 'lr' or 'both'\n");
	mexPrintf("       for Up-Down, Left-Right or Both.\n\n");
	mexPrintf("       cvlib_mex('flip',IMG,'DIR');\n");
	mexPrintf("       Flips IMG inplace.\n");
}

/* -------------------------------------------------------------------------------------------- */
void filterUsage() {
	mexPrintf("Usage: B = cvlib_mex('filter',IMG,FILTER);\n");
	mexPrintf("       Convolves image IMG with the kernel FILTER\n\n");

	mexPrintf("       Class support: uint8, int16, uint16 or single.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG and 1 copy of FILTER.\n");
}

/* -------------------------------------------------------------------------------------------- */
void smoothUsage() {
	mexPrintf("Usage: B = cvlib_mex('smooth',IMG [,TYPE,PAR1,PAR2,PAR3,PAR4]);\n");
	mexPrintf("       Smooths the MxNx3 or MxN IMG in one of several ways\n");
	mexPrintf("       The available TYPE of smoothing are:\n");
	mexPrintf("       'blur' summation over a pixel PAR1�PAR2 neighborhood with subsequent scaling by 1/(PAR1xPAR2).\n");
	mexPrintf("       'gaussian' convolves image with PAR1�PAR2 Gaussian kernel [Default].\n");
	mexPrintf("       'median' finding median of PAR1�PAR1 neighborhood (i.e. the neighborhood is square)\n");
	mexPrintf("       'bilateral' apply bilateral 3x3 filtering with color sigma=PAR1 and space sigma=PAR2\n");
	mexPrintf("	          If PAR1 & PAR2 are not provided they default to 5 and 50.\n");
	mexPrintf("	          Information about bilateral filtering can be found at.\n");
	mexPrintf("	          www.dai.ed.ac.uk/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html\n");
	mexPrintf("	  PAR1 -> The first parameter of smoothing operation [Default = 5]\n");
	mexPrintf("	  PAR2 -> The second parameter of smoothing operation [Default = PAR1]\n");
	mexPrintf("	  PAR3 -> In case of Gaussian kernel this parameter may specify Gaussian sigma\n");
	mexPrintf("	          (standard deviation). If it is zero, it is calculated from the kernel size\n");
	mexPrintf("	  PAR4 -> In case of non-square Gaussian kernel the parameter may be used to specify\n");
	mexPrintf("	          a different (from param3) sigma in the vertical direction.\n\n");
	mexPrintf("       B = cvlib_mex('smooth',IMG);\n");
	mexPrintf("           Does a gaussian filtering with a 5x5 kernel.\n");

	mexPrintf("       Class support: logical, uint8 or single.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG.\n");
}

/* -------------------------------------------------------------------------------------------- */
void pyrUUsage() {
	mexPrintf("Usage: B = cvlib_mex('pyrU',IMG);\n");
	mexPrintf("       performs up-sampling step of Gaussian pyramid decomposition.\n");
	mexPrintf("       First it upsamples the source image by injecting even zero rows and\n");
	mexPrintf("       and then convolves result with a 5x5 Gaussian filter multiplied by 4\n");
	mexPrintf("       for interpolation. So the destination image is four times larger than\n");
	mexPrintf("       the source image.\n\n");

	mexPrintf("       Class support: logical, uint8, uint16, int16, single or double.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG and 1 copy of B.\n");
}

/* -------------------------------------------------------------------------------------------- */
void pyrDUsage() {
	mexPrintf("Usage: B = cvlib_mex('pyrD',IMG);\n");
	mexPrintf("       performs downsampling step of Gaussian pyramid decomposition.\n");
	mexPrintf("       First it convolves source image with a 5x5 Gaussian filter and then\n");
	mexPrintf("       downsamples the image by rejecting even rows and columns\n\n");

	mexPrintf("       Class support: logical, uint8, uint16, int16, single or double.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG and 1 copy of B.\n");
}

/* -------------------------------------------------------------------------------------------- */
void arithmUsage() {
	mexPrintf("Usage: B = cvlib_mex(OP,IMG1,IMG2);\n");
	mexPrintf("       Apply per element arithmetics betweem IMG1 & IMG2. OP can be one of:\n");
	mexPrintf("       add -> B = IMG1 + IMG2\n");
	mexPrintf("       sub -> B = IMG1 - IMG2\n");
	mexPrintf("       mul -> B = IMG1 * IMG2\n");
	mexPrintf("       div -> B = IMG1 / IMG2\n\n");
	mexPrintf("       If IMG2 is a scalar than OP can also take these values:\n");
	mexPrintf("       addS -> B = IMG1 + IMG2\n");
	mexPrintf("       subS -> B = IMG1 - IMG2\n\n");

	mexPrintf("The form (that is, with no output)\n: cvlib_mex(OP,IMG1,IMG2);\n");
	mexPrintf("       does the above operation in-place and stores the result in IMG1\n\n");

	mexPrintf("       Class support: all but uint32.\n");
	mexPrintf("       Memory overhead: none.\n");
}

/* -------------------------------------------------------------------------------------------- */
void addWeightedUsage() {
	mexPrintf("Usage: B = cvlib_mex('addweighted',IMG1,alpha,IMG2,beta[,gamma]);\n");
	mexPrintf("       The function addweighted calculates weighted sum of two arrays as following:\n");
	mexPrintf("       B(i) = img1(i)*alpha + img2(i)*beta + gamma\n");
	mexPrintf("       If gamma is not provided, gamma = 0\n\n");
	mexPrintf("The form (that is, with no output)\n: cvlib_mex('addweighted',IMG1,alpha,IMG2,beta[,gamma]);\n");
	mexPrintf("       does the above operation in-place and stores the result in IMG1\n\n");

	mexPrintf("       Class support: all but uint32.\n");
	mexPrintf("       Memory overhead: none.\n");
}

/* -------------------------------------------------------------------------------------------- */
void inpaintUsage() {
	mexPrintf("Usage: cvlib_mex('inpaint',IMG1,IMG2);\n");
	mexPrintf("       The function inpaint reconstructs IMG1 image area from the pixel near the area boundary.\n");
	mexPrintf("       The function may be used to remove scratches or undesirable objects from images:\n");
	mexPrintf("       IMG1 is a uint8 MxNx3 rgb OR a MxN intensity image and IMG2 is a mask array\n");
	mexPrintf("       of the same size (MxN) of IMG1. It can be a logical array or a uint8 array.\n");
	mexPrintf("       IMG = cvlib_mex('polyline',IMG1,IMG2);\n");
	mexPrintf("       Returns the drawing in the the new array IMG.\n\n");

	mexPrintf("       Class support: uint8.\n");
	mexPrintf("       Memory overhead: 1 copy of IMG and a MxN matrix of the same type as IMG.\n");
}
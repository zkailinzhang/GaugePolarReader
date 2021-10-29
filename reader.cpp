
#include "reader.hpp"



Mat Reader::readimg(string src_image_path){
         	
	        this->srcimg = imread(src_image_path, 1);
            return this->srcimg.clone();
     };

void Reader::readimg(Mat &src_img,Vec4i x){
            Mat zhizhen = src_img(cv::Rect(int(x[0])-15,int(x[1])-15,int(x[2])-int(x[0])+30,int(x[3])-int(x[1])+30));
            this->srcimg = zhizhen;
     };

Vec3d Reader::avg_circles(vector<Vec3f> circles, int b){
        int avg_x=0;
        int avg_y=0;
        int avg_r=0;
        for (int i=0;  i< b; i++ )
            {
            //平均圆心 半径
            avg_x = avg_x + circles[i][0];
            avg_y = avg_y + circles[i][1];
            avg_r = avg_r + circles[i][2];
            }
        //半径为啥int
        avg_x = int(avg_x/(b));
        avg_y = int(avg_y/(b));
        avg_r = int(avg_r/(b));
        
        Vec3d xyr = Vec3d(avg_x, avg_y, avg_r);
	    return xyr; 
        
    }

float Reader::getDist_P2L(cv::Point2f pointP, cv::Point2f pointA, cv::Point2f pointB)
{
    float A = 0, B = 0, C = 0;
    A = pointA.y - pointB.y;
    B = pointB.x - pointA.x;
    C = pointA.x*pointB.y - pointA.y*pointB.x;

    float distance = 0.0;
    distance = ((float)abs(A*pointP.x + B*pointP.y + C)) / ((float)sqrtf(A*A + B*B));
    return distance;
}


float Reader::dist_2_pts(int x1, int y1, int x2, int y2){
        float pp = pow(x2-x1,2)+pow(y2-y1,2);
        float tmp = sqrt(pp);
        return tmp;
        }

Mat Reader::region_of_interest(Mat img,vector<vector<Point>> vertices){
    Mat mask = Mat::zeros(img.size(), img.type());
    int match_mask_color= 255;

    fillPoly(mask, vertices, Scalar(match_mask_color));
    imwrite("mask.jpg", mask);

    Mat masked_image;
    bitwise_and(img, mask,masked_image);
    imwrite("masked_image.jpg", masked_image);
    return masked_image;


}

void Reader::maxtwosqens(vector<float> s,int& sta,int& end){


    vector<int> s2=s;
    int j = 0 ;
    int max = 0 ;
    int qi=0;
    int zhong=0;
        
    for(int i = 0;i<s.size();i++){
        if(s[i] == 0){
            j++;
            zhong=i;    
        }
        else{
            if(j>max){
                max = j;
                zhong = i-1;
            }
            j = 0 ;
        }
    }
        
    if(j>max)max = j ;
      
    int qirow = zhong-max;   
    int j1 = 0 ;
    int max1 = 0 ;
    int qi1=0;
    int zhong1=0;
        
    for(int i = 0;i<s.size();i++){
        if(s[i] == 0){
                j1++;
                zhong1=i;    
        }
        else{
            if(j1>max1){
                max1 = j1;
                zhong1 = i-1;
            }
            j1 = 0 ;
            }
    }
        
    if(j1>max1)max1 = j1 ;

        
    int qirow1 = zhong1-max1; 
    int startke=0;
    int endke =0;
        
    if (qirow1>qirow){   
        startke=zhong1-max1;
        endke =zhong;
    }
    else{
        startke=zhong+1;
        endke =zhong1-max1;   
    }
        

    sta =startke;
    end = endke;
}


float Reader::polardetect(){
Mat midd_img = this->srcimg.clone();
        int wight = midd_img.rows;
        int height = midd_img.cols;
        // int col = sumcol.cols;
        // int row = sumcol.rows;
        Mat gray_img;
        cvtColor(midd_img, gray_img, COLOR_BGR2GRAY);

        Mat canny;
        Canny(gray_img,canny,200,50);


        int thresh =120;
        int maxValue = 255;
        Mat midd_img2;
        threshold(gray_img,midd_img2, thresh, maxValue, CV_THRESH_BINARY_INV);

        Mat lin_polar_img;
        linearPolar(midd_img2, lin_polar_img, Point2f(this->x,this->y), this->r, CV_WARP_FILL_OUTLIERS + CV_INTER_LINEAR);

        int pwight = lin_polar_img.rows;
        int pheight = lin_polar_img.cols;

        //什么类型，float int
        Mat sumcol;
        reduce(lin_polar_img, sumcol, 1, CV_REDUCE_SUM);

        vector<float> sumcolvec ;


        for(int i=0;i< sumcol.rows;i++){
            //要不要加0
            int p=sumcol.at<float>(i,0);
            sumcolvec.push_back(p);
        }

        //min_angle = *(max_element(sumcolvec.begin(), sumcolvec.end())+1);
        vector<float>::iterator maxPosition = max_element(sumcolvec.begin(), sumcolvec.end());
        int posi = maxPosition - sumcolvec.begin();
        //min_angle = frth_angle_[maxPosition - frth_sub.begin()+1];
        int zhenpos=0;
        int qipos=0;
        int zhipos=0;
        int zhenpos=int(posi);


        float separation = 10.0 ;
        int interval = int(360 / separation);
        
        Mat p3 = cv::Mat::zeros(cv::Size(interval,2),CV_32FC1); 
        vector<Point> pts;

        for(int i =0;i<interval;i++){
            Point pp;
            for(int j=0; j<2;j++){
                if (j%2==0)
                    pp.x=this->x + 1.0 * r * cos(separation * i * CV_PI / 180);
                else
                    pp.y=this->y + 1.0 * r * sin(separation * i * CV_PI / 180);
            }
            pts.push_back(pp);
        }

        Mat canny;
        Canny(gray_img,canny,200,50);
        //Mat region_of_interest_vertices= p3;
        imwrite("canny.jpg", canny);
       
        vector<vector<Point>> region_of_interest_vertices;
        region_of_interest_vertices.push_back(pts);
        
        

        Mat cropped_image= region_of_interest(canny, region_of_interest_vertices);
        imwrite("cropped_image.jpg", cropped_image);

        Mat maskpl=Mat::zeros(cropped_image.size(),CV_8UC1);
        Mat contours3= midd_img.clone();

        vector<vector<Point>> contours;  
        vector<Vec4i> hierarchy;  
        findContours(cropped_image,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_NONE); 

        vector<vector<cv::Point> > int_cnt;
        for(int i=0;i<contours.size();i++)  
        {  
                float area =contourArea(contours[i]);
                Rect prect  = boundingRect(contours[i]);

                float cpd = dist_2_pts(prect.x+prect.width/2,prect.y+prect.height/2,this->x, this->y);

                if ((area <500) && (cpd< this->r*4/4) && (cpd > this->r*3/4)){

                    drawContours(contours3, vector<vector<Point> >(1,contours[i]), -1, Scalar(255,0,0), 3);
                    drawContours(maskpl, vector<vector<Point> >(1,contours[i]), -1, 255, 3);

                    int_cnt.push_back(contours[i]) ;
                }
        }  
        Mat polar6;
        linearPolar(maskpl, polar6, Point2f(this->x,this->y), this->r, CV_WARP_FILL_OUTLIERS + CV_INTER_LINEAR);


        vector<float> sum6t;
        vector<float> sum6;
        vector<vector<float>> sum6tt;

        for (int col=0;col< pwight ;col++){

            Mat col1 = polar6.colRange(col,col+1);
            for (int i=0;i<pheight-1;){
                float p1=col1.at<float>(i,0);
                float p2=col1.at<float>(i,0);    
                float p21 = abs(p2-p1);
                i+=2;
                sum6t.push_back(p21);
            }
            sum6tt.push_back(sum6t);

            int suum=accumulate(sum6t.begin(),sum6t.end(),0); 
            sum6.push_back(suum);

        }

        vector<float>::iterator maxPosition = max_element(sum6.begin(), sum6.end());
        int posi = maxPosition - sumcolvec.begin();
        
        vector<float>  kedulist = sum6tt[posi];

        Mat polar63;
        cvtColor(polar6, polar63, COLOR_BGR2GRAY);

        cv::line(polar63, Point(int(posi-3), int(0)), Point(int(posi-3),int(pwight)),Scalar(255, 0,255 ), 3);

        vector<int> list1;
        vector<int> list2;
        vector<int> list3;
        int possum=0;

        int startke=0,endke=0;
        maxtwosqens(kedulist,startke,endke);
        startke *=2;
        endke *=2;

        cv::line(polar63, Point( int(0),int(endke)), Point(int(pwight),int(endke)),Scalar(255, 0,255 ), 3);
        cv::line(polar63, Point( int(0),int(startke)),Point(int(pwight),int(startke)),Scalar(255, 0,255 ), 3);



        qipos,zhipos=startke,endke;
        float dushu = (zhenpos - qipos)/(pheight-qipos+zhipos)*(1.0-0.0);

        return dushu;
}


Mat Reader::detectCircles(){

        Mat midd_img = this->srcimg.clone();
        int wight = midd_img.rows;
        int height = midd_img.cols;

        Mat gray_img;
        cvtColor(midd_img, gray_img, COLOR_BGR2GRAY);
        medianBlur(gray_img, gray_img, 5);

        vector<Vec3f> circles;
        
        // HoughCircles(gray_img, circles, HOUGH_GRADIENT, 1,
        //     gray_img.rows / 16,     // change this value to detect circles with different distances to each other
        //     100, 30, 127, 138		// change the last two parameters
        //                             // (min_radius & max_radius) to detect larger circles
        // );
        //HoughCircles(gray_img, circles,cv2.HOUGH_GRADIENT, 1, 120,  100, 50, int(height*0.35), int(height*0.48));
        HoughCircles(gray_img, circles,HOUGH_GRADIENT, 1, 120,  100, 50, int(height*0.35), int(height*0.48));
        
        int b = circles.size();
        Vec3d xyr = this->avg_circles(circles, b);

        this->x=xyr[0];
        this->y=xyr[1];
        this->r=xyr[2];

        //画圆和圆心
        circle(midd_img, Point(this->x, this->y), this->r, (0, 0, 255), 3,LINE_AA);
        circle(midd_img, Point(this->x, this->y), 2, (0, 255, 0), 3, LINE_AA);
        
        imwrite("jianceyuan.jpg", midd_img);


        Mat imgtt= midd_img.clone();
       
        float separation = 10.0 ;
        int interval = int(360 / separation);
        //p1 = np.zeros((interval,2))  
        vector<Point> p1;
        vector<Point> p2;
        vector<Point> p_text;
        //p_text = np.zeros((interval,2))
        for (int i=0;i<interval ;i++){
            Point pp;
            for(int j=0; j<2;j++){
                if (j%2==0){
                    pp.x = this->x + 0.88 * r * cos(separation * i * CV_PI / 180);
                    }
                else{
                    pp.y  = this->y + 0.88 * r * sin(separation * i * CV_PI / 180);
                   }
            }
            p1.push_back(pp);
        }


        int text_offset_x = 10;
        int text_offset_y = 5;
        for (int i=0;i<interval ;i++){
            Point pp, p_t;
            for(int j=0; j<2;j++){
                if (j%2==0){
                    pp.x = this->x + r * cos(separation * i * 3.14 / 180);
                    p_t.x= this->x - text_offset_x + 1.2 * r * cos((separation) * (i+9) * 3.14 / 180); //point for text labels, i+9 rotates the labels by 90 degrees
                }else{
                    pp.y = this->y + r * sin(separation * i * 3.14 / 180);
                    p_t.y= this->y + text_offset_y + 1.2* r * sin((separation) * (i+9) * 3.14 / 180);//point for text labels, i+9 rotates the labels by 90 degrees
                }
            }
            p2.push_back(pp);
            p_text.push_back(p_t);

        }

        for(int i=0;i<interval;i++){
            cv::line(imgtt, Point(int(p1[i].x), int(p1[i].y)), Point(int(p2[i].x), int(p2[i].y)),Scalar(0, 255, 0), 2);
            putText(imgtt, to_string(int(i*separation)), Point(int(p_text[i].x), int(p_text[i].y)), FONT_HERSHEY_SIMPLEX, 0.3,Scalar(0,0,0),1,LINE_AA);
        }

        imwrite("calibrate.jpg", imgtt);


        //separation=10;
        //interval = int(360/separation);
        
        Mat p3 = cv::Mat::zeros(cv::Size(interval,2),CV_32FC1); 
        vector<Point> pts;

        for(int i =0;i<interval;i++){
            Point pp;
            for(int j=0; j<2;j++){
                if (j%2==0)
                    pp.x=this->x + 1.0 * r * cos(separation * i * CV_PI / 180);
                else
                    pp.y=this->y + 1.0 * r * sin(separation * i * CV_PI / 180);
            }
            pts.push_back(pp);
        }

        Mat canny;
        Canny(gray_img,canny,200,50);
        //Mat region_of_interest_vertices= p3;
        imwrite("canny.jpg", canny);
       
        vector<vector<Point>> region_of_interest_vertices;
        region_of_interest_vertices.push_back(pts);
        
        

        Mat cropped_image= region_of_interest(canny, region_of_interest_vertices);
        
        imwrite("cropped_image.jpg", cropped_image);

        Mat contours3= midd_img.clone();

        vector<vector<Point>> contours;  
        vector<Vec4i> hierarchy;  

        //findContours(cropped_image,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point());  
        findContours(cropped_image,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_NONE); 

        //Mat imageContours=Mat::zeros(image.size(),CV_8UC1);  
        //Mat Contours=Mat::zeros(image.size(),CV_8UC1);  //绘制 
        //std::vector<int> int_cnt;
        vector<vector<cv::Point> > int_cnt;

        for(int i=0;i<contours.size();i++)  
        {  
                float area =contourArea(contours[i]);
                Rect prect  = boundingRect(contours[i]);

                float cpd = dist_2_pts(prect.x+prect.width/2,prect.y+prect.height/2,this->x, this->y);

                if ((area <500) && (cpd< this->r*4/4) && (cpd > this->r*3/4)){

                    drawContours(contours3, vector<vector<Point> >(1,contours[i]), -1, Scalar(255,0,0), 3);
                    
                    int_cnt.push_back(contours[i]) ;
                }
        }  
        imwrite("contours3.jpg", contours3);
        

        //10 350
        float reference_zero_angle= 20;
        float reference_end_angle= 340;
        float min_angle=90;
        float max_angle=270;

        std::vector<int> frth_quad_index;
        std::vector<int> thrd_quad_index;
        std::vector<float> frth_quad_angle;
        std::vector<float> thrd_quad_angle;

        for(int i =0;i<int_cnt.size();i++){
                    //contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数  
            vector<cv::Point> conPoints;
            float x1,y1;
            float sx1=0,sy1=0;
            for(int j=0;j<contours[i].size();j++)   
            {  
                //绘制出contours向量内所有的像素点  
                //Point P=Point(contours[i][j].x,contours[i][j].y); 
                //conPoints.push_back(P);
                sx1+=contours[i][j].x;
                sy1+=contours[i][j].y;
            }
            x1 = sx1/contours[i].size();
            y1 = sy1/contours[i].size();

            float  xlen= x1 - this->x;
            float  ylen= this->y - y1;

            //double res = atan2(float(ylen), float(xlen));
            //res = res * 180.0 / M_PI;

            if ((xlen<0) &&(ylen<0)){
                double res = atan2(abs(float(ylen)), abs(float(xlen)));
                res = res * 180.0 / M_PI; 
                float final_start_angle= 90-res;
                
                frth_quad_index.push_back(i);
                frth_quad_angle.push_back(final_start_angle);

                if (final_start_angle> reference_zero_angle)
                    if (final_start_angle<min_angle)
                        min_angle= final_start_angle;
            
            }

            if ((xlen>0) &&(ylen<0)){
                double res = atan2(abs(float(ylen)), abs(float(xlen)));
                res = res * 180.0 / M_PI; 
                float final_end_angle= 270+res;
                
                thrd_quad_index.push_back(i);
                thrd_quad_angle.push_back(final_end_angle);

                if (final_end_angle< reference_end_angle)
                    if (final_end_angle>max_angle)
                        max_angle= final_end_angle;
            }
        
        }

        cout<<"Zero reading corresponds to:"<< min_angle<<endl;
        cout<<"End reading corresponds to:"<< max_angle<<endl;

        vector<float> frth_angle_(frth_quad_angle);
        vector<float> thrd_angle_(thrd_quad_angle);
        
        //升序 ，，降序
        std::sort(frth_angle_.begin(), frth_angle_.end(), std::less<float>());
        std::sort(thrd_angle_.begin(), thrd_angle_.end(), std::greater<float>());
        
        vector<float> frth_sub; 
        vector<float> thrd_sub;
        for (int i=0;i<frth_angle_.size()-1;i++)
            frth_sub.push_back(frth_angle_[i+1]-frth_angle_[i]);
        for (int i=0;i<thrd_angle_.size()-1;i++)
            thrd_sub.push_back(thrd_angle_[i+1]-thrd_angle_[i]);


        vector<float>::iterator maxPosition = max_element(frth_sub.begin(), frth_sub.end());
        min_angle = frth_angle_[maxPosition - frth_sub.begin()+1];
        //min_angle = *(max_element(frth_sub.begin(), frth_sub.end())+1);

        vector<float>::iterator minPosition = min_element(thrd_sub.begin(), thrd_sub.end());
        max_angle = thrd_angle_[minPosition - thrd_sub.begin()+1];
        
        this->min_angle =min_angle;
        this->max_angle =max_angle;
        cout<<"Zero reading corresponds to:"<< this->min_angle<<endl;
        cout<<"End reading corresponds to:"<< this->max_angle<<endl;

	    return midd_img;
    };

float Reader::detectLines(){
        Mat gray_img;
        Mat midd_img = this->srcimg.clone();
        cvtColor(midd_img, gray_img, COLOR_BGR2GRAY);
        //50cm 模糊3像素
        cv::Ptr<cv::CLAHE>  clahe = createCLAHE(40.0, Size(8,8));
        Mat dstcle;
        //限制对比度的自适应阈值
        clahe->apply(gray_img,dstcle);
        //原图一定屏蔽掉，模糊的要添加，原图添加，识别不了， 模糊的 不添加 识别不了
        //gray2 =dst
        Mat dst_img;

        int thresh =166;

        int maxValue = 255;
        Mat midd_img2;
        
        //Canny(gray_img, midd_img2, 23, 55, 3);

        // convert cannied image to a gray one  
        //cvtColor(midd_img2, dst_img, CV_GRAY2BGR);

        // define a vector to collect all possible lines 
        vector<Vec4i> mylines;
        int g_nthreshold = 39;
        
        //minLineLength = 10
        //maxLineGap = 0
        //image=dst2, rho=3, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=0
        //HoughLinesP(midd_img, mylines, 1, CV_PI / 180, g_nthreshold + 1, 20, 5);
        imwrite("gray_img.jpg", gray_img);
        threshold(gray_img,midd_img2, thresh, maxValue, CV_THRESH_BINARY_INV);
        imwrite("midd_img2.jpg", midd_img2);
        HoughLinesP(midd_img2, mylines, 3, CV_PI / 180, 100, 10, 0);
        //HoughLinesP(gray_img, mylines, 3, CV_PI / 180, 100, 10, 0);
        
        Point circle_center = Point2f(this->x,this->y);
        float circle_radius = this->r;

        cout<<"circle_center:"<< circle_center<<endl;
        cout<<"circle_radius:"<< circle_radius<<endl;
        // draw every line by using for
        Mat midd_img3 = midd_img.clone();
        for (size_t i = 0; i < mylines.size(); i++)
        {
            Vec4i l = mylines[i];
            //直线是四元组，直线的起点坐标 终点坐标，
            //起点 与圆心 xy坐标 10以内
            //终点 在圆边以内，
            if (((circle_center.x - 10) <= l[0]) && (l[0] <= (circle_center.x + 10)))
                if (((circle_center.y - 10) <= l[1]) && (l[1] <= (circle_center.y + 10)))
                    if (((circle_center.x - circle_radius) <= l[2]) && (l[2] <= (circle_center.x + circle_radius)))
                        if (((circle_center.y - circle_radius) <= l[3]) && (l[3] <= (circle_center.y + circle_radius)))
                        {
                            //cout << Point(l[0], l[1]) << " " << Point(l[2], l[3]) << " " << l[0] << " " << circle_center.x - circle_radius << " " << circle_center.x + circle_radius << endl;
                            cv::line(midd_img3, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(23, 180, 55), 2, CV_AA);
                            Vec4i cho_l = l;
                            //cv::line(midd_img3, Point(circle_center.x, circle_center.y), Point(cho_l[2], cho_l[3]), Scalar(23, 180, 55), 2, CV_AA);
                        }
        }
        imwrite("midd_img3.jpg", midd_img3);
        float diff1LowerBound = 0.0;
        float diff1UpperBound = 0.25;
        float diff2LowerBound = 0.0 ;
        float diff2UpperBound = 1.0;
        
        float r = circle_radius;
        vector<Vec4i> final_line_list;
        vector<float> distance_list;
        vector<float> line_length_list;
        Mat midd_img6 = midd_img.clone();

        for (size_t i =0;i<mylines.size();i++){
            Vec4i l = mylines[i];
            float diff1 = this->dist_2_pts(circle_center.x,circle_center.y,l[0],l[1]);
            float diff2 = this->dist_2_pts(circle_center.x,circle_center.y,l[2],l[3]);
            if (diff1 > diff2){
                    float temp = diff1;
                    diff1 = diff2;
                    diff2 = temp;}
            
            if (((diff1<diff1UpperBound*r) && (diff1>diff1LowerBound*r)) && ((diff2<diff2UpperBound*r) && (diff2>diff2LowerBound*r))){
                    
                    float line_length = this->dist_2_pts(l[0],l[1], l[2],l[3]);
                    float distance =  getDist_P2L(Point2f(x,y),Point2f(l[0],l[1]),Point2f( l[2],l[3]));
                    
                    // if ((line_length>0.1*r)  && (distance>-20) && (distance <10)){
                         final_line_list.push_back(Vec4i(l[0],l[1], l[2],l[3]));
                         distance_list.push_back(distance);
                        line_length_list.push_back(line_length);
                        cv::line(midd_img6, Point(l[0],l[1]), Point(l[2],l[3]), Scalar(23, 180, 55), 2, CV_AA);
                    // }
                    
                    }

        };
        imwrite("midd_img6.jpg", midd_img6);

        //输出第一个线，点到直线的距离，点到两个端点的距离，线的长度；最短距离的位置，线最长的位置
        vector<float>::iterator maxPosition = max_element(line_length_list.begin(), line_length_list.end());
        vector<float>::iterator minPosition = min_element(distance_list.begin(), distance_list.end());
        
        Vec4i final_line;
        cout<<"maxPosition "<<maxPosition - line_length_list.begin()+1<<endl;

        if (maxPosition == minPosition){
            final_line = final_line_list[maxPosition - line_length_list.begin()+1];
        }
        else{
            final_line = final_line_list[maxPosition - distance_list.begin()+1];
        }

        float x1 = final_line[0];
        float y1 = final_line[1];
        float x2 = final_line[2];
        float y2 = final_line[3];
        Mat midd_img7 = midd_img.clone();
        cv::line(midd_img7, Point(x1, y1), Point(x2, y2), Scalar(23, 180, 55), 3,CV_AA);
        imwrite("midd_img7.jpg", midd_img7);


        //find the farthest point from the center to be what is used to determine the angle
        float dist_pt_0 = this->dist_2_pts(circle_center.x, circle_center.y, x1, y1);
        float dist_pt_1 = this->dist_2_pts(circle_center.x, circle_center.y, x2, y2);

        float x_angle=0.0;
        float y_angle=0.0;
        if (dist_pt_0 > dist_pt_1){
            x_angle = x1 - x;
            y_angle = y - y1;
            }
        else{
            x_angle = x2 - x;
            y_angle = y - y2;
        }

        x_angle = (x1+x2)/2 - x;
        y_angle = y - (y1+y2)/2;


        double res = atan2(float(y_angle), float(x_angle));

        //these were determined by trial and error
        res = res * 180.0 / M_PI;
        
        float final_angle =0.0;

        if ((x_angle > 0) && (y_angle > 0))//in quadrant I
            final_angle = 270 - res;
        if (x_angle < 0 && y_angle > 0) //in quadrant II
            final_angle = 90 - res;
        if (x_angle < 0 && y_angle < 0)  //in quadrant III
            final_angle = 90 - res;
        if (x_angle > 0 && y_angle < 0)  //in quadrant IV
            final_angle = 270 - res;

        cout<<"this->Config->min_angle "<<min_angle<<endl;
        cout<<"this->Config->max_angle "<<max_angle<<endl;
        
        vector<float> final_value_list;
        for(int i=0;i<10;i++){

            float old_min = float(this->min_angle)+i-5;
            float old_max = float(this->max_angle)+i-5;

            float new_min = float(min_value);
            float new_max = float(max_value);

            float old_value = final_angle;

            float old_range = (old_max - old_min);
            float new_range = (new_max - new_min);
            float final_value = (((old_value - old_min) * new_range) / old_range) + new_min;
            final_value_list.push_back(final_value);
        }
        
        float dushu = accumulate(final_value_list.begin() , final_value_list.end() , 0.0);
        dushu = dushu/float(final_value_list.size()) ;
        
        cout<<"final_value "<<dushu<<endl;
        return dushu;

    };
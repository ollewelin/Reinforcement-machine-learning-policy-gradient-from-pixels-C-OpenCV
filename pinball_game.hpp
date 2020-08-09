#ifndef PINBALL_GAME_H
#define PINBALL_GAME_H
///Add advanced game avoid rectangles but catch circles
///Fix bugg in pinball_game.hpp (was work on raspierry pi but not on PC) replay_count was not set to 0 at init state. Now replay_count=0;
///Increase size of angle character and sign of angle so the Agent cas see this rezized char and sign +/- then it seems to learn that.

#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur
#include <stdio.h>
///#include <raspicam/raspicam_cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <cstdlib>
#include <ctime>
#include <math.h>  // exp
#include <stdlib.h>// exit(0);
#include <iostream>
#include <opencv2/core/types_c.h>
using namespace std;
using namespace cv;

class pinball_game
{
public:
    pinball_game()///Constructor
    {
        printf("Construct a arcade game object\n");
    }
    virtual ~pinball_game()///Destructor
    {
        printf("Destruct game object\n");
    }
    ///== User settings used outside this object but Set by this object =============
    int use_character;///1=enable charackters
    int enable_ball_swan;///1= enbale swan after the ball
    int use_image_diff;
    int high_precition_mode; ///This will make adjustable rewards highest at center of the pad.
    int use_dice_action;
    int drop_out_percent;
    int Not_dropout;
    int flip_reward_sign;
    int print_out_nodes;
    int replay_times;///If =0 no replay. >0 this is the nuber of replay with serveral diffrent actions so the ageint take the best rewards before make any weights update
    int replay_count;//
    int replay_episode;
    float pix2hid_learning_rate;
    float hid2out_learning_rate;
    float max_rewards;
    int enabel_3_state;///1= then enable 3 actions UP, DOWN, STOP
    int advanced_game;///0= only a ball. 1= ball give awards. square gives punish

    bool use_unfair_dice;
    int rand_nr_of_frames_change_unfair_dice;
    int change_unfair_dice_frame_cnt;
    float random_unfair_dice_bias;//Change every time counter change_unfair_dice_frame_cnt cleared
    float random_unfair_dice_gain;//

    ///=== End user settings ========
    int game_Width;///Pixel width of game grapics. A constant value set in init_game
    int game_Height;///Pixel height of game grapics. A constant value set in init_game
    int move_up;///Input Action from Agent. 1= Move up pad. 0= Move down pad. 2= STOP used only when enabel_3_state = 1
    int win_this_game;///1= Catched the ball this episode. 0= miss. This will be used as the reward feedback to the Reinforcment Learning
    int pad_ball_diff;
    int square;///Always 1 when advanced_game = 0. In advanced_game = 1 mode the this ball could be =0 the a square is writed instad of a ball
    int nr_of_frames;///The number of frames on one episode. A constant value set in init_game
    int slow_motion;///1= slow down speed of game 0= full speed
    Mat gameGrapics;///This is the grapics of the game how is the enviroment
    int episode;///Episode is only use to ensure no pattern in randomizer
    void init_game(void);
    void start_episode(void);
    void run_episode(void);
    void set_user_settings(void);
    int frame;
protected:
private:
    int pad_position;
    int ball_pos_x;
    int ball_pos_y;
    float ball_angle_derivate;///Example 0 mean strigh forward. +1.0 mean ball go up 1 pixel on one frame 45 deg.
    float save_replay_start_ball_ang;
    int frame_steps;
    int ball_offset_y;///
    ///int ball;///Always 1 when advanced_game = 0. In advanced_game = 1 mode the this ball could be =0 the a square is writed instad of a ball
    CvPoint P1;///The ball point OpenCV
    CvPoint P2;///The pad point uppe corner OpenCV
    CvPoint P3;///The pad point lower corner OpenCV

};


void pinball_game::set_user_settings(void)
{
    char answer_character;
    getchar();
    printf("Do you want to use the differance between last pre and now image as input <Y>/<N> \n");
    printf("Example use <N>\n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        use_image_diff=1;
    }
    else
    {
        use_image_diff=0;
    }
    getchar();
    printf("Do you print out output node values <Y>/<N> \n");
    printf("Example use <N>\n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        print_out_nodes=1;
    }
    else
    {
        print_out_nodes=0;
    }
    getchar();

    printf("Do you want LOW precition mode (only -1 ot +4 arwards)  <Y>/<N> \n");
    printf("Example use <N>\n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        high_precition_mode=0;
    }
    else
    {
        high_precition_mode=1;///Now the rewards be depend on how good centerd the ball was catched.
    }
    getchar();

    printf("Do you want to use Action made from dice AND policy network <Y>/<N> \n");
    printf("Example use <Y>\n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        use_dice_action=1;
    }
    else
    {
        use_dice_action=0;
    }
    getchar();
    printf("Do you want to use dropout at hidden node <Y>/<N> \n");
    printf("Example use <N>\n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        Not_dropout=0;
        printf("Enter (int) drop_out_percent (drop_out_percent =%d)\n", drop_out_percent);
        scanf("%d", &drop_out_percent);
    }
    else
    {
        Not_dropout=1;
        drop_out_percent=0;
    }
    getchar();
    printf("Do you want flip sign of Awarwd  <Y>/<N> \n");
    printf("Example use <N>\n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        flip_reward_sign =1;
    }
    else
    {
        flip_reward_sign =0;
    }
    getchar();
    printf("Do you want to enable swan after the ball  <Y>/<N> \n");
    printf("Example use <Y>\n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        enable_ball_swan =1;
    }
    else
    {
        enable_ball_swan =0;
    }
    getchar();

    printf("Do you want advanced game ball give awards square give punish  <Y>/<N> \n");
    printf("Example use <Y>\n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        advanced_game =1;
    }
    else
    {
        advanced_game =0;
    }
    getchar();

    printf("Do you want to enable extra characters on the graphics  <Y>/<N> \n");
    printf("Example use <Y>\n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        use_character =1;
    }
    else
    {
        use_character =0;
    }
    getchar();
    printf("Do you want to use default learining rate and max rewards parameters  <Y>/<N> \n");
    printf("Example use <Y>\n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        ///Default Learning rate
        printf("Default learning rate and default max rewards used\n");
    }
    else
    {
        printf("Enter (float) pix2hid_learning_rate (default was =%f)\n", pix2hid_learning_rate);
        scanf("%f", &pix2hid_learning_rate);
        printf("Enter (float) hid2out_learning_rate (default was =%f)\n", hid2out_learning_rate);
        scanf("%f", &hid2out_learning_rate);
        printf("Enter (float) max_rewards (default was =%f)\n", max_rewards);
        scanf("%f", &max_rewards);
    }

    getchar();
    printf("Do you want to use replay same ball direction several times and only use bets rewards <Y>/<N> \n");
    printf("Example use <N>\n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        printf("Enter (int) replay_times (default was =%d)\n", replay_times);
        scanf("%d", &replay_times);
     }
    else
    {
        replay_times = 0;
        printf("No replay. replay_times =%d\n", replay_times);
    }

    getchar();
    printf("Do you want to enable 3 state action UP, DOWN and STOP <Y>/<N> \n");
    printf("Example use <N>\n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        enabel_3_state =1;
    }
    else
    {
        enabel_3_state =0;
    }
    getchar();

    printf("Do you want to enable unfair dice <Y>/<N> \n");
    printf("Example use <N>\n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        use_unfair_dice =1;
        printf("Enter (float) random_unfair_dice_gain (default was =%f)\n", random_unfair_dice_gain);
        scanf("%f", &random_unfair_dice_gain);
    }
    else
    {
        use_unfair_dice =0;
    }
    getchar();

    printf("********* You have enter this settings ***********\n");
    printf("pix2hid_learning_rate =%f \n", pix2hid_learning_rate);
    printf("hid2out_learning_rate =%f \n", hid2out_learning_rate);
    printf("use_unfair_dice =%d \n", use_unfair_dice);
    printf("random_unfair_dice_gain =%f \n", random_unfair_dice_gain);
    printf("max_rewards =%f \n", max_rewards);
    printf("Not_dropout =%d \n", Not_dropout);
    printf("drop_out_percent =%d \n", drop_out_percent);
    printf("use_character =%d \n", use_character);
    printf("enable_ball_swan =%d \n", enable_ball_swan);
    printf("use_image_diff =%d\n", use_image_diff);
    printf("high_precition_mode =%d\n", high_precition_mode);
    printf("use_dice_action =%d\n", use_dice_action);
    printf("flip_reward_sign =%d\n", flip_reward_sign);
    printf("print_out_nodes =%d\n", print_out_nodes);
    printf("enabel_3_state =%d\n", enabel_3_state);
    printf("replay_times =%d\n", replay_times);
    printf("********* Good luck with your settings :) ***********\n");
}

void pinball_game::init_game(void)
{
    //print opencv version
    printf("opencv version: %d.%d.%d\n",CV_VERSION_MAJOR,CV_VERSION_MINOR,CV_VERSION_REVISION);

    replay_count=0;
    use_character=0;
    enable_ball_swan=0;///default yes swan
    slow_motion=0;
    move_up=0;///Init down if there was no Agent action done.
    win_this_game=0;///Init
    frame=0;///Init with frame 0 for the episode
    pad_position = game_Height/2;///Start the game at center
    game_Width = 220;///
    game_Height = 200;///
    nr_of_frames = 100;///
    gameGrapics.create(game_Height, game_Width, CV_32FC1);
    gameGrapics = Scalar(0.0f);///Init with Black
    srand (static_cast <unsigned> (time(0)));///Seed the randomizer
    ball_angle_derivate = (float) (rand() % 65535) / 65536;///Set ball (here only first time) shoot angle. Random value 0..1.0 range
    use_unfair_dice = 0;
    rand_nr_of_frames_change_unfair_dice = nr_of_frames / 2;
    change_unfair_dice_frame_cnt = 0;
    random_unfair_dice_bias = 0.0f;//Change every time counter change_unfair_dice_frame_cnt cleared
    random_unfair_dice_gain = 0.6f;//

}

void pinball_game::start_episode(void)
{
    if(replay_episode ==0)
    {
        if(replay_count < replay_times)
        {
            replay_count++;
            ball_angle_derivate = save_replay_start_ball_ang;
        }
        else
        {
            replay_count = 0;
            ball_angle_derivate = (float) (rand() % 65535) / 65536;///Set ball shoot angle. Random value 0..1.0 range
            save_replay_start_ball_ang = ball_angle_derivate;
        }
    }
    else
    {
        ball_angle_derivate = save_replay_start_ball_ang;
    }
    ball_angle_derivate *= 6.0;
    ball_angle_derivate -= 3.0;/// -0.5..+0.5 will mean +/- 12.5 deg random ball angle
    frame_steps=0;
    ball_offset_y = game_Height/2;///
    pad_position = game_Height/2;///Start the game at center
    if(advanced_game == 1)
    {
        float ball_or_square =0.0;
        ball_or_square = (float) (rand() % 65535) / 65536;///Set ball shoot angle. Random value 0..1.0 range
        if(ball_or_square > 0.5f)
        {
            square = 1;
        }
        else
        {
            square = 0;
        }
    }
    else
    {
        square = 0;
    }
}

void pinball_game::run_episode(void)
{
    int circle_size = 4;
    int rect_size = 14;
    int rect_x_expand = 15;///Expand rectangle x direction
    int ball_start_x = 10;///Start 10 pixel inside game plan
    int y_bounce_constraints = 20;
    int y_pad_constraints = 28;
    int pad_width = 3;
    int pad_height = 40;
    int pad_speed = 4;//4


///The frame loop is outside this class so The Agient cad do actions each frame step

    frame_steps++;///This is need to handle bounce. This will reset when bounce
    ///=========== Draw the ball =================
    ///Bounce handle
    ball_pos_y = ((int) ((float)frame_steps * ball_angle_derivate)) + ball_offset_y;///

    if(ball_pos_y > (game_Height-y_bounce_constraints) || ball_pos_y < y_bounce_constraints)
    {
        frame_steps=0;
        ball_angle_derivate = -ball_angle_derivate;
        float bounce_extra;
        if(ball_angle_derivate<0.0f)
        {
            bounce_extra = -1.0f;///This ensure bounce even if ball_angle_derivate is less then +/-1.0 angle
        }
        else
        {
            bounce_extra = 1.0f;///This ensure bounce even if ball_angle_derivate is less then +/-1.0 angle
        }
        ball_offset_y = ball_pos_y + (ball_angle_derivate+bounce_extra);
    }
    ball_pos_x = (frame * 2) + ball_start_x;///Take 2 pixel step forward
    P1.x = ball_pos_x;///Set to control grapic OpenCV circle() below
    P1.y = ball_pos_y;///Set to control grapic OpenCV circle() below
    if(enable_ball_swan==1)
    {

        float *index_ptr_gameGapics = gameGrapics.ptr<float>(0);

        for(int i=0; i<gameGrapics.rows * gameGrapics.cols; i++)
        {
            *index_ptr_gameGapics = *index_ptr_gameGapics * 0.85f;
            index_ptr_gameGapics++;
        }
        P2.y = 30;
        P3.y = 50;
        P2.x = 0;
        P3.x = 20;
        rectangle(gameGrapics, P2, P3, Scalar(0.0), 15);/// Erease characters
        P2.y = 0;
        P3.y = 20;
        P2.x = 0;
        P3.x = 20;
        rectangle(gameGrapics, P2, P3, Scalar(0.0), 15);/// Erease characters

        P2.y = 0;
        P3.y = game_Height;
        P2.x = game_Width-10;
        P3.x = game_Width-10;
        rectangle(gameGrapics, P2, P3, Scalar(0.0), 2);///  Erase old pad

    }
    else
    {
        gameGrapics = Scalar(0.05f);///Begin with all black then draw up grapics ball and pad
    }

    if(square == 0)
    {
        ///Ball
        circle(gameGrapics, P1, circle_size, Scalar(0.9), 7, -1);///Ball size 7
    }
    else
    {
        ///Rectangle in advanced_game
        P2.y = ball_pos_y - (rect_size);
        P3.y = ball_pos_y + (rect_size);
        P2.x = ball_pos_x - (rect_size)-rect_x_expand;
        P3.x = ball_pos_x + (rect_size)+rect_x_expand;
        rectangle(gameGrapics, P2, P3, Scalar(0.9), 3);///Square
    }
    ///===============================

    ///========= Draw the Pad ==============

    if(!(frame > nr_of_frames-2))
    {


        if(pad_position > (game_Height-y_pad_constraints))
        {
            ///Allow only move up

            if(move_up==1)
            {
                pad_position = pad_position - pad_speed;
            }
            else if(move_up==0)
            {
                pad_position = (game_Height-y_pad_constraints) + pad_speed;
            }
            else
            {
                ///STOP move only used when enable_3_state = 1
            }

        }
        else if(pad_position < y_pad_constraints)
        {
            ///Allow only move down
            if(move_up==0)
            {
                pad_position = pad_position + pad_speed;
            }
            else if(move_up==0)
            {
                pad_position = (y_pad_constraints) - pad_speed;
            }
            else
            {
                ///STOP move only used when enable_3_state = 1
            }
        }
        else
        {
            ///Move up or down
            if(move_up==1)
            {
                pad_position = pad_position - pad_speed;
            }
            else if(move_up==0)
            {
                pad_position = pad_position + pad_speed;
            }
            else
            {
                ///STOP move only used when enable_3_state = 1
            }
        }
    }
    P2.y = pad_position - (pad_height/2);
    P3.y = pad_position + (pad_height/2);
    P2.x = - (pad_width/2) + game_Width-10;
    P3.x = (pad_width/2) + game_Width-10;
    rectangle(gameGrapics, P2, P3, Scalar(0.8), 2);///   C++: void rectangle(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
    if(frame > nr_of_frames-2)
    {
        ///This episode is over
        ///Is the pad catch the ball ??
        if(((pad_position + (pad_height/2)) < ball_pos_y) || ((pad_position - (pad_height/2)) >  ball_pos_y))
        {
            ///Miss
            if(square == 1){
            win_this_game = 1;
            printf("Win \n");
            }
            else{
               win_this_game = 0;
                printf("Lose \n");
            }
        }
        else
        {
            ///Catch
            if(square == 1){
               win_this_game = 0;
                printf("Lose \n");
             }
             else
             {
                win_this_game = 1;
                printf("Win \n");
            }
        }
        pad_ball_diff = pad_position - ball_pos_y;
    }

    if(use_character==1)
    {

        char episode_char = ((char) episode);
        char ball_ang_char;
        string angle = "xx";
        std::string::iterator It = angle.begin();
        if(ball_angle_derivate < 0.0)
        {
            *It = '-';
            ball_ang_char = (char) (-ball_angle_derivate*4.2);

        }
        else
        {
            *It = '+';
            ball_ang_char = (char) (ball_angle_derivate*4.2);

        }
        It++;
        ///ball_ang_char *= 10;
        *It = ball_ang_char+48;
///        cv::putText(gameGrapics, angle, cvPoint((5+episode_char/20),(175+(char) (rand() % 16))), CV_FONT_HERSHEY_PLAIN, 2, cvScalar(55),2);
        //cv::putText(gameGrapics, angle, cvPoint((5),(175)), CV_FONT_HERSHEY_PLAIN, 4, cvScalar(0.5),3);// CV_... for Opencv3.1
        cv::putText(gameGrapics, angle, cvPoint((5),(175)), FONT_HERSHEY_PLAIN, 4, cvScalar(0.5),3);// CV_FONT_HERSHEY_PLAIN dont work in opencv4
        ///  char rand_char = rand() % 255;
        string random = "x";
        std::string::iterator It2 = random.begin();

        if(episode_char < 40)
        {
            episode_char = episode_char + 40;
        }

        *It2 = episode_char ;///Episode is only use to ensure no pattern in randomizer
        //cv::putText(gameGrapics, random, cvPoint((3+episode_char/20),(35+((char) (rand() % 16)))), CV_FONT_HERSHEY_PLAIN, 2, cvScalar(0.5),2);// CV_... for Opencv3.1
        cv::putText(gameGrapics, random, cvPoint((3+episode_char/20),(35+((char) (rand() % 16)))), FONT_HERSHEY_PLAIN, 2, cvScalar(0.5),2);
    }

    imshow("Game", gameGrapics);
    if(slow_motion==1)
    {
        waitKey(20);///Wait 100msec
    }
    else
    {
        waitKey(1);///Wait 1msec for only OpenCV grapics
    }
}

#endif // PINBALL_GAME_H

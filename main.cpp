///Add advanced game avoid rectangles but catch circles
///Fix bugg in pinball_game.hpp (was work on raspierry pi but not on PC) replay_count was not set to 0 at init state. Now replay_count=0;
/// Only dependancy is OpenCV C++ library need to be installed
/// Update with gamma parameter now
/// Example of Reinforced Machine Learning attached on a simple Pinball game
/// The enviroment (enviroment = data feedback) for the Agient (Agient = machine learning system)
/// is the raw pixels 50x50 pixels (2500 input nodes) and 200 hidden nodes on 100 frames
/// So the input to hidden weights is 50x50x100x200 x4 bytes (float) = is 200Mbytes huges but it work anyway!!
///Enhancment to do in future.
///TODO: Add some layers of Convolutions (with unsupervised Learning for learning feature patches) will probably enhance preformance.
///TODO: Maybe add bias weigth is a good idee to enhance preformance or stability during training.
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/imgproc/imgproc.hpp> //
#include <stdio.h>
///#include <raspicam/raspicam_cv.h> //Only for raspberry pi
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <cstdlib>///rand() srand()
#include <ctime>
#include <math.h>  // exp
#include <stdlib.h>// exit(0);
#include <iostream>
using namespace std;
using namespace cv;
#include "pinball_game.hpp"


#include <omp.h>///Use of Multi thread add -fopenmp at C::B : Settings->Compiler..->Compiler Options->Other Options and at: Settings->Compiler..Linker Settings->Other Linker Options
///#define MULTI_THRED_4X

#include <unistd.h>
void clearScreen()
{
    const char *CLEAR_SCREEN_ANSI = "\e[1;1H\e[2J";
    write(STDOUT_FILENO, CLEAR_SCREEN_ANSI, 12);
}

#include <random>
double dice_mean = 0.5;
double dice_stddev = 1.3;
double dice_stddev_start = 1.3;
double dice_stddev_decrease = 0.03;
double dice_minimum_stddev = 0.20;
int dice_dec_stddev_after_nr_episodes = 1000;


double gaussian_dice(int number_of_examples, int example_nr, bool print_examples)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    // std::default_random_engine generator;
    std::normal_distribution<double> distribution(dice_mean, dice_stddev);
    double number = 0.0;
    if (print_examples == false)
    {
        int nr_of_try = 100;//If never inside 0.0..1.0 during this number of trys skip this round and give value 0.5
        for(int i=0;i<nr_of_try;i++){
            number = distribution(generator);//run the guassian dice
            if(number>=0.0 && number<=1.0){
                break;//The gaussian dice is inside range. Return this number
            }
            else{
                if(i==nr_of_try-1){
                    number = 0.5;//Give up and return 0.5 if nr_of_try ends;
                    printf("Give up gaussion dice after %d try, outside range 0.0..1.0 return default value = %f\n", i, (float)number);
                }
            }
        }
    }
    else
    {
        printf("%d examples of a gaussian dice this will be used as an unfair dice\n", number_of_examples);
        printf("Example number %d\n", example_nr+1);
        const int nrolls = 1000; // number of experiments
        const int nstars = 300;  // maximum number of stars to distribute

        const int stars_array_size = 10;
        float p[stars_array_size] = {};
        for (int i = 0; i < stars_array_size; ++i)
        {
            p[i] = 0.0;
        }
        for (int i = 0; i < nrolls; ++i)
        {
            number = distribution(generator);
            int numberXarrSize = int(number * stars_array_size);
            if ((number >= 0.0) && (number < 1.0))
                ++p[numberXarrSize];
            double numberTimeArraySize = numberTimeArraySize * (double)stars_array_size;
        }
        printf("Dice gaussian distribution (0.0,1.0):\n");
        for (int i = 0; i < stars_array_size; ++i)
        {

            float one_tenth = ((float)i) / stars_array_size;
            printf("%1.1f", one_tenth);
            printf("..");
            printf("%1.1f", (one_tenth + (1.0 / (double)stars_array_size)));
            printf(": ");
            std::cout << std::string(p[i] * nstars / nrolls, '*') << std::endl;
        }
    }
    return number;
}

const float Relu_neg_gain = 0.01f;///A small positive value here will prevent ReLU neuoron from dying. 0.0f pure rectify (1.0 full linear = nothing change)
float relu(float input)
{
    float output=0;
    output = input;
    if(input < 0.0)
    {
        output = input * Relu_neg_gain;
    }
    return output;
}
int int_abs_value(int signed_value)
{
    int abs_v;
    abs_v = signed_value;
    if(abs_v < 0)
    {
        abs_v = -abs_v;
    }
    return abs_v;
}
float revers_sigmoid(float sigm)
{
    float prevent_infinity = 0.001f;
    float rev_sigm;
    ///y = 1/(1+exp(-x)) ///Sigmoid
    ///x = -log((1-y)/y) ///Sigmoid revers log = ln
    if(sigm > (1.0f-prevent_infinity))
    {
        sigm = (1.0f-prevent_infinity);
    }
    if(sigm < prevent_infinity)
    {
        sigm = prevent_infinity;
    }
    rev_sigm = -log((1.0f-sigm)/sigm);
    return rev_sigm;
}
void randomize_dropoutHid(int *zero_ptr_dropoutHidden, int HiddenNodes, int verification, int drop_out_percent)
{
    int drop_out_part = HiddenNodes * drop_out_percent/100;//
    int*ptr_dropoutHidden;

    for(int i=0; i<HiddenNodes; i++)
    {
        ptr_dropoutHidden = zero_ptr_dropoutHidden + i;
        *ptr_dropoutHidden = 0;//reset
    }
    int check_how_many_dropout = 0;
    if(verification == 0)
    {
        for(int k=0; k<HiddenNodes*2; k++) ///Itterate max HiddenNodes*2 number of times then give up to reach drop_out_part
        {
            for(int i=0; i<(drop_out_part-check_how_many_dropout); i++)
            {
                int r=0;
                r = rand() % (HiddenNodes-1);
                ptr_dropoutHidden = zero_ptr_dropoutHidden + r;
                *ptr_dropoutHidden = 1;///
            }
            check_how_many_dropout = 0;
            for(int j=0; j<HiddenNodes; j++)
            {
                ptr_dropoutHidden = zero_ptr_dropoutHidden + j;
                check_how_many_dropout += *ptr_dropoutHidden;
            }
            if(check_how_many_dropout >= drop_out_part)
            {
                break;
            }
        }
        //  printf("check_how_many_dropout =%d\n", check_how_many_dropout);
    }
}


#define check_win_prob_ittr 1000
void printGaussianDiceSettings(void)
{
    printf("**************** Gaussian dice settings: ******************\n");
    printf("mean,               dice_mean =   %f\n", dice_mean);
    printf("standard deviation, dice_stddev = %f\n", dice_stddev);
    printf("***********************************************************\n");
}
int main()
{
    int nrOfExamples = 45;
    for(int i=0;i<nrOfExamples;i++)
    {
        clearScreen();
        printf("\n");
        printf("\n");
        printf("i=%d\n",i);
        dice_stddev -= dice_stddev_decrease;
        if (dice_stddev < dice_minimum_stddev)
        {
            dice_stddev = dice_minimum_stddev;
        }
        printGaussianDiceSettings();
        gaussian_dice(nrOfExamples, i, true);
        waitKey(500);
    }

    for(int i=0;i<nrOfExamples;i++)
    {
        double testDice = gaussian_dice(nrOfExamples, i, false);
        printf("Test gaussian dice = %f\n", (float)testDice);
        waitKey(100);
    }

    dice_stddev = dice_stddev_start; 
    printGaussianDiceSettings();
    gaussian_dice(1, 0, true);

    float random_f;
    char filename[100];
    char answer_character;
    printf("Reinforcment Learning test of pixels data input from a simple ping/pong game\n");

    FILE *fp2;
    int nr_of_episodes=0;
    int auto_save_w_counter =100;
    const int auto_save_after = 2000;///Auto Save weights after this number of episodes
    int show_w_counter =19;///Show the weights graphical not ever episodes (to save CPU time)
    const int show_w_after = 20;///Show the weights graphical not ever episodes (to save CPU time)
    int pixel_height = 50;///The input data pixel height, note game_Width = 220
    int pixel_width = 50;///The input data pixel width, note game_Height = 200
    Mat resized_grapics, test, pix2hid_weight, hid2out_weight;
    Size size(pixel_width,pixel_height);//the dst image size,e.g.100x100
    pinball_game gameObj1;///Instaniate the pinball game
    gameObj1.init_game();///Initialize the pinball game with serten parametrers
    gameObj1.slow_motion=0;///0=full speed game. 1= slow down
    gameObj1.replay_times = 0;///If =0 no replay. >0 this is the nuber of replay with serveral diffrent actions so the ageint take the best rewards before make any weights update
    gameObj1.advanced_game = 1;///0= only a ball. 1= ball give awards. square gives punish
    float pix2hid_learning_rate = 0.4f;///0.02
    float hid2out_learning_rate = 0.1f;///0.001
/*
    if(gameObj1.replay_times > 0)
    {
        pix2hid_learning_rate *= (float)gameObj1.replay_times;///0.02
        hid2out_learning_rate *= (float)gameObj1.replay_times;///0.001
    }
*/
    ///========== Setup weights and nodes for the Reinforcemnt learning network ==========================
    ///This will contain all the training weigts for training. It will have exact same size as the grapics * number of frames * hidden nodes
    ///Use also here OpenCV Mat so it is easy to Visualize some of the data as well as store the weights
    int Nr_of_hidden_nodes = 40;///Number of hidden nodes on one frame weight
    float *input_node_stored;///This will have a record of all frames of the resized_grapics used for weight updates
    input_node_stored = new float[pixel_width * pixel_height * gameObj1.nr_of_frames];
    int visual_nr_of_frames = 20;///Visualization weights of a only few frames othewize the image will be to large
    int visual_nr_of_hid_node = 20;///Visualization weights of a only few hidden nodes othewize the image will be to large
    pix2hid_weight.create(pixel_height * visual_nr_of_hid_node, pixel_width * visual_nr_of_frames, CV_32FC1);///Visualization weights of a only few frames and hidden nodes othewize the image will be to large CV_32FC1 is pixel format
    float *pix2hid_weightB;///File data read/write connect to tied weights
    pix2hid_weightB = new float[pixel_height * pixel_width * Nr_of_hidden_nodes * gameObj1.nr_of_frames];///File data is same size as all tied weights pix2hid_weight.create
    float *hidden_node;
    hidden_node = new float[gameObj1.nr_of_frames * Nr_of_hidden_nodes];///200 hidden nodes and 100 frames for example
    float *hidden_delta;
    hidden_delta = new float[gameObj1.nr_of_frames * Nr_of_hidden_nodes];///200 hidden nodes and 100 frames for example
    hid2out_weight.create(gameObj1.nr_of_frames, Nr_of_hidden_nodes, CV_32FC1);///Use also here OpenCV Mat so it is easy to Visualize the data as well as store weights
    float *hid2out_weightB;
    hid2out_weightB = new float[gameObj1.nr_of_frames * Nr_of_hidden_nodes];
    float *output_node;
    output_node = new float[gameObj1.nr_of_frames];
    float *output_delta;
    output_delta = new float[gameObj1.nr_of_frames];
    float *action;
    action = new float[gameObj1.nr_of_frames];
    float *best_actions;///Used for pick and save the best rewards action serie to then replay that event serie and only update weight from that action serie. used when replay_times > 0
    best_actions = new float[gameObj1.nr_of_frames];
    int *dropoutHidden;///dropout table
    dropoutHidden = new int[gameObj1.nr_of_frames * Nr_of_hidden_nodes];///data 0 normal fc_hidden_node. 1= dropout this fc_hidden_node this training turn.
    

    ///Some reports to user
    printf("use_unfair_dice = %d\n", gameObj1.use_unfair_dice);
    printf("Number of hidden nodes to one frames = %d\n", Nr_of_hidden_nodes);
    printf("Total number of hidden nodes fo all frames together = %d\n", gameObj1.nr_of_frames * Nr_of_hidden_nodes);
    printf("Number of output nodes alway equal to the number of frames on one episode = %d\n", gameObj1.nr_of_frames);
    ///===================================================================================================

    ///=================== index variable for the weights ====================
    int ix=0;///index to f_data[ix]
    ///=======================================================================

    ///============ Prepare pointers to make it possible to direct acces Mat data matrix ==================
    test = gameObj1.gameGrapics.clone();
    resize(test, resized_grapics, size);
    /// only used when use_diff ==1
        Mat diff_grap, pre_grap;
        diff_grap = resized_grapics.clone();
        pre_grap = resized_grapics.clone();
        float *zero_ptr_diff_grap = diff_grap.ptr<float>(0);///zero_... Always point at first pixel
        float *index_ptr_diff_grap = diff_grap.ptr<float>(0);///index_... Adjusted to abritary pixel




    float *zero_ptr_res_grap = resized_grapics.ptr<float>(0);///zero_... Always point at first pixel
    float *index_ptr_res_grap = resized_grapics.ptr<float>(0);///index_... Adjusted to abritary pixel
    float *zero_ptr_pix2hid_w = pix2hid_weight.ptr<float>(0);///Only used for visualization of weights
    float *index_ptr_pix2hid_w = pix2hid_weight.ptr<float>(0);///Only used for visualization of weights
    float *zero_ptr_hid2out_w = hid2out_weight.ptr<float>(0);///Only used for visualization of weights
    float *index_ptr_hid2out_w = hid2out_weight.ptr<float>(0);///Only used for visualization of weights
    ///====================================================================================================

    ///================ Initialize weight with random noise =====================================
    printf("Insert noise to weights. Please wait...\n");
    srand (static_cast <unsigned> (time(0)));///Seed the randomizer (need to do only once)
    float start_weight_noise_range = 0.05f;///0.05
    float Rando=0.0f;
    for(int i=0; i<(pixel_height * pixel_width * Nr_of_hidden_nodes * gameObj1.nr_of_frames); i++)
    {
        Rando = (float) (rand() % 65535) / 65536;//0..1.0 range
        Rando -= 0.5f;
        Rando *= start_weight_noise_range;
        pix2hid_weightB[i] = Rando;///Insert the noise to the weight pixel to hidden
    }
    printf("Noise to the weight pixel to hidden is inserted\n");
    for(int i=0; i<(gameObj1.nr_of_frames * Nr_of_hidden_nodes); i++)
    {
        Rando = (float) (rand() % 65535) / 65536;//0..1.0 range
        Rando -= 0.5f;
        Rando *= start_weight_noise_range;
        hid2out_weightB[i] = Rando;
    }
    printf("Noise to the weight hidden to output node is inserted\n");
    ///==================== End of Initialize weight with random noise ===========================

    ///============ Regardning Load weights to file ==========================

    gameObj1.use_image_diff=0;
    gameObj1.high_precition_mode = 1; ///This will make adjustable rewards highest at center of the pad.
    gameObj1.use_dice_action=1;
    gameObj1.drop_out_percent=0;
    gameObj1.Not_dropout=1;
    gameObj1.flip_reward_sign =0;
    gameObj1.print_out_nodes = 0;
    float gamma = 0.97f;
    gameObj1.enable_ball_swan =1;
    gameObj1.use_character =1;
    gameObj1.max_rewards = 5.0f;
    gameObj1.enabel_3_state = 0;
    gameObj1.pix2hid_learning_rate = pix2hid_learning_rate;
    gameObj1.hid2out_learning_rate = hid2out_learning_rate;
    int best_rewards_serie=0;
    float highest_rewards=-1000.0f;
    int use_noise_image=0;///This is only if you want to add noise on input graphics to test the systems how it handle noise
    float noise_thres = 0.15;///This is only if you use noise on input graphics to test the systems how it handle noise
///
    printf("Would you like to add noise on input image <Y>/<N> \n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        printf("********** Add image noise **********\n");
        use_noise_image=1;
    }
    else
    {
        printf("********** No image noise **********\n");
        use_noise_image=0;
    }
    getchar();
///
    printf("Would want to use default settings <Y>/<N> \n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        printf("********** Default settings **********\n");
        getchar();
    }
    else
    {
        printf("********** Set user settings **********\n");
        gameObj1.set_user_settings();
        pix2hid_learning_rate = gameObj1.pix2hid_learning_rate;
        hid2out_learning_rate = gameObj1.hid2out_learning_rate;
    }
    printf("pix2hid_learning_rate = %f\n", pix2hid_learning_rate);
    printf("hid2out_learning_rate = %f\n", hid2out_learning_rate);
    printf("Would you like to load stored weights, pix2hid_weight.dat and hid2out_weight.dat <Y>/<N> \n");
    printf("Example use (if you don't already trainied some good files) <N>\n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
     sprintf(filename, "pix2hid_weight.dat");
     //  sprintf(filename, "/media/pi/USB DISK2/pix2hid_weight.dat");
        fp2 = fopen(filename, "r");
        if (fp2 == NULL)
        {
            printf("Error while opening file pix2hid_weight.dat");
            exit(0);
        }
        printf("Start so load pix2hid_weight.dat Please wait... The file size is = %d bytes\n", (sizeof pix2hid_weightB[0]) * (pixel_height * pixel_width * Nr_of_hidden_nodes * gameObj1.nr_of_frames));
        fread(pix2hid_weightB, sizeof pix2hid_weightB[0], (pixel_height * pixel_width * Nr_of_hidden_nodes * gameObj1.nr_of_frames), fp2);///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
        fclose(fp2);
        printf("weights are loaded from pix2hid_weight.dat file\n");
        sprintf(filename, "hid2out_weight.dat");
        fp2 = fopen(filename, "r");
        if (fp2 == NULL)
        {
            printf("Error while opening file hid2out_weight.dat");
            exit(0);
        }
        printf("Start so load hid2out_weight.dat Please wait... The file size is = %d bytes\n", (sizeof hid2out_weightB[0]) * (gameObj1.nr_of_frames * Nr_of_hidden_nodes));
        fread(hid2out_weightB, sizeof hid2out_weightB[0], (gameObj1.nr_of_frames * Nr_of_hidden_nodes), fp2);
        fclose(fp2);
        printf("weights are loaded from hid2out_weight.dat file\n");
    }
    ///============ End of Regardning Load weights to file ==========================

///test and debug
float threshold_1;
float threshold_2;
threshold_1 = (1.0f/3.0f);
threshold_2 = (2.0f/3.0f);
printf("threshold_1 %f\n", threshold_1);
printf("threshold_2 %f\n", threshold_2);
float rev_out_node;
float rev_rand_dice;
float action_dice;


float win_probability = 0.0f;
float win_lose_sum = 0.0f;
int win_lose_array[check_win_prob_ittr];
int win_lose_cnt = 0;

    while(1)
    {
        
        gameObj1.replay_episode = 0;
        float dot_product = 0.0f;
        gameObj1.start_episode();///Staring a new game turn
    //    float dice=0;///Only random for the first frame
        randomize_dropoutHid(&dropoutHidden[0], (gameObj1.nr_of_frames * Nr_of_hidden_nodes), gameObj1.Not_dropout, gameObj1.drop_out_percent);///select dropout node to the hidden node
        for(int frame_g=0; frame_g<gameObj1.nr_of_frames; frame_g++) ///Loop throue each of the 100 frames
        {

            output_node[frame_g] = 0.0f;///Start with clear this node
            gameObj1.frame = frame_g;
            gameObj1.run_episode();
            test = gameObj1.gameGrapics.clone();
            resize(test, resized_grapics, size);
            if(gameObj1.use_image_diff==1)
            {
                if(frame_g==0)
                {
                    pre_grap = Scalar(0.0f);
                }
                else
                {
                    diff_grap = pre_grap - resized_grapics;
                    pre_grap = resized_grapics.clone();///Used to calculate diff_grap For next frame
                }
                //diff_grap = diff_grap + 0.5f;
            }
            ///=============== Forward data for this frame ==================
            ///Make the Dot product to this frames hidden nodes and output node

            for(int i=0; i<Nr_of_hidden_nodes; i++)
            {
                hidden_node[frame_g * Nr_of_hidden_nodes + i] = 0.0f;///Start with clear this value before sum up the dot product

                dot_product = 0.0f;///Start with clear this value before sum up the dot product
                for(int j=0; j<(pixel_height * pixel_width); j++)
                {
                    ix = ((pixel_width * Nr_of_hidden_nodes) * (pixel_height * frame_g + j/pixel_width) + (pixel_width * i) + j%pixel_width);///Prepare the index to point on the right place in the weight matrix pix2hid_weightB[]
                    if(gameObj1.use_image_diff==1)
                    {
                        index_ptr_diff_grap = zero_ptr_diff_grap + j;///Prepare the pointer address to point on the right place on the grapical image of this grapical frame
                        dot_product += pix2hid_weightB[ix] * (*index_ptr_diff_grap);///Make the dot product of Weights * Game grapichs
                        input_node_stored[pixel_width * pixel_height * frame_g + j] = (*index_ptr_diff_grap);///Save this grame grapich pixel must read this pixel later when update weights
                    }
                    else
                    {
                        index_ptr_res_grap = zero_ptr_res_grap + j;///Prepare the pointer address to point on the right place on the grapical image of this grapical frame
                        dot_product += pix2hid_weightB[ix] * (*index_ptr_res_grap);///Make the dot product of Weights * Game grapichs
                        if(use_noise_image==1 && i==0)
                        {
                            float noise;
                            noise = ((float) (rand() % 65535) / 65536) ;
                            if(noise < noise_thres)
                            {
                               (*index_ptr_res_grap) = ((float) (rand() % 65535) / 65536);
                            }
                        }
                        input_node_stored[pixel_width * pixel_height * frame_g + j] = (*index_ptr_res_grap);///Save this grame grapich pixel must read this pixel later when update weights
                    }
                }

                ///Relu this dot product
                if(dropoutHidden[(frame_g * Nr_of_hidden_nodes + i)] == 0)
                {
                    ///Normal forward not drop out this node
                    dot_product = relu(dot_product);
                }
                else
                {
                    dot_product = 0.0f;
                }
                hidden_node[frame_g * Nr_of_hidden_nodes + i] = dot_product;///Put this finnish data in the hidden node neuron
                ix = (frame_g * Nr_of_hidden_nodes + i);
                output_node[frame_g] += hid2out_weightB[ix] * dot_product;///Take this hidden node data how is for the moment => hidden_node[frame_g * Nr_of_hidden_nodes + i] = dot_product;
            }
            output_node[frame_g] = 1.0/(1.0 + exp(-(output_node[frame_g])));///Sigmoid function.  x = 1.0/(1.0 + exp(-(x)))
            ///=============== End Forward data for this frame ==================

            imshow("resized_grapics", resized_grapics);///  resize(src, dst, size);
            if(gameObj1.use_image_diff==1)
            {
                imshow("diff", diff_grap);
            }
            waitKey(1);


            ///=============== Made the action =======
            ///Update 2017-09-02 9:30 ===== Use dice with the policy network output probability for what action should be done
            ///float action_dice;
            if(gameObj1.use_dice_action == 1)
            {
                action_dice = (float) (rand() % 65535) / 65536;///Update 2017-09-02 9:30 ===== Use dice with the policy network output probability for what action should be done
            }
            else
            {
                action_dice = 0.5f;///Old test mode
            }
            if(gameObj1.use_unfair_dice == true)
            {
/*
                if(frame_g == 0)
                {
                    //First randomazie how offen the unfair dice will change unfair bias this eposode
                    gameObj1.rand_nr_of_frames_change_unfair_dice = (int) (((float) (rand() % 65535) / 65536) * (float) gameObj1.nr_of_frames);
                    gameObj1.change_unfair_dice_frame_cnt = 0;
                    //printf("rand_nr_of_frames_change_unfair_dice = %d\n", gameObj1.rand_nr_of_frames_change_unfair_dice);
                }
                if(gameObj1.change_unfair_dice_frame_cnt < gameObj1.rand_nr_of_frames_change_unfair_dice){
                    gameObj1.change_unfair_dice_frame_cnt++;}
                else{
                    gameObj1.change_unfair_dice_frame_cnt = 0;
                }
                if(gameObj1.change_unfair_dice_frame_cnt == 0){
                    //Change unfair dice bias
                    float half_unfair_gain = gameObj1.random_unfair_dice_gain * 0.5f;
                    gameObj1.random_unfair_dice_bias = ((float) (rand() % 65535) / 65536) * gameObj1.random_unfair_dice_gain;
                    gameObj1.random_unfair_dice_bias = gameObj1.random_unfair_dice_bias - half_unfair_gain;
                  //  printf("gameObj1.random_unfair_dice_bias =%f\n", gameObj1.random_unfair_dice_bias);
                }
                action_dice = action_dice + gameObj1.random_unfair_dice_bias;
                //Truncate dice between 0..1
                if(action_dice > 1.0f){
                    action_dice = 1.0f;
                }
                if(action_dice < 0.0f){
                    action_dice = 0.0f;
                }
*/
                action_dice = (float)gaussian_dice(0,0,false);
                //printf("action_dice =%f\n", action_dice);


            }
            if(gameObj1.enabel_3_state ==1)
            {
                printf("3 state mode removed exit program\n");
                exit(0);
            }
            else
            {
                /// 2 state mode UP, DOWN
                ///Use the data from probablility from policy network
                if(output_node[frame_g] > action_dice)
                {
                    action[frame_g] = 1.0f;
                    gameObj1.move_up = 1;
                }
                else
                {
                    action[frame_g] = 0.0f;
                    gameObj1.move_up = 0;
                }
            }
            ///================= End Action ==================

        }
        if(gameObj1.print_out_nodes==1)
        {
            for(int i=0; i<gameObj1.nr_of_frames; i++)
            {
                printf("output_node[frame nr %d] = %f\n", gameObj1.nr_of_frames-1-i, output_node[gameObj1.nr_of_frames-1-i]);
            }

        }

        ///========== Auto save weights to files ====================
        if(auto_save_w_counter>auto_save_after)
        {
            auto_save_w_counter=0;
            sprintf(filename, "pix2hid_weight.dat");
            fp2 = fopen(filename, "w+");
            if (fp2 == NULL)
            {
                printf("Error while opening file pix2hid_weight.dat");
                exit(0);
            }
            printf("Start so save pix2hid_weight.dat Please wait... The file size is = %d bytes\n", (sizeof pix2hid_weightB[0]) * (pixel_height * pixel_width * Nr_of_hidden_nodes * gameObj1.nr_of_frames));
            fwrite(pix2hid_weightB, sizeof pix2hid_weightB[0], (pixel_height * pixel_width * Nr_of_hidden_nodes * gameObj1.nr_of_frames), fp2);
            fclose(fp2);

            printf("weights are saved at hid2out_weight.dat file\n");
            sprintf(filename, "hid2out_weight.dat");
            fp2 = fopen(filename, "w+");
            if (fp2 == NULL)
            {
                printf("Error while opening file hid2out_weight.dat");
                exit(0);
            }
            printf("Start so save hid2out_weight.dat Please wait... The file size is = %d bytes\n", (sizeof hid2out_weightB[0]) * (gameObj1.nr_of_frames * Nr_of_hidden_nodes));
            fwrite(hid2out_weightB, sizeof hid2out_weightB[0], (gameObj1.nr_of_frames * Nr_of_hidden_nodes), fp2);
            fclose(fp2);
            printf("weights are saved at hid2out_weight.dat file\n");
        }
        else
        {
            auto_save_w_counter++;
        }
        ///========== End Auto save weights to files ====================

        ///========== Show visualization of the weights not nessesary ==========
        if(show_w_counter>show_w_after)
        {
            show_w_counter=0;
            for(int i=0; i<(gameObj1.nr_of_frames * Nr_of_hidden_nodes); i++)
            {
                ///Visualization of all hid2out weights
                index_ptr_hid2out_w = zero_ptr_hid2out_w + i;
                (*index_ptr_hid2out_w) = hid2out_weightB[i] + 0.5f;///(*index_ptr_hid2out_w) is the pointer to Mat hid2out_weight. +0.5 make a grayscale with gray center 0.5;
            }
            ///int start_show_frame = gameObj1.nr_of_frames - visual_nr_of_frames-1;///Show only the last (20 = visual_nr_of_frames) frame weights
            int show_ever5frame_start =0;
            for(int si=0; si<(pixel_height * visual_nr_of_hid_node * pixel_width * visual_nr_of_frames); si++) ///si = visualize Mat itterator
            {
                ///Visualize pix2hid weight reagrding only few frames and few hidden nodes connections
                ///pix2hid_weight.create(pixel_height * visual_nr_of_hid_node, pixel_width * visual_nr_of_frames, CV_32FC1);///
                ///The pix2hid_weightB[ix] is organized like this;
                ///ix = ((pixel_width * Nr_of_hidden_nodes) * (pixel_height * frame_g + j/pixel_width) + (pixel_width * i) + j%pixel_width);
                ///where..
                ///j  is itterator for (pixel_height * pixel_width) is the pixel area of the game (shrinked are rezised image)
                ///i  is itterator for Nr_of_hidden_nodes
                ///frame_g is of course the frame number
                int vis_colum = 0;
                vis_colum = si%(pixel_width * visual_nr_of_hid_node);
                int vis_row = 0;
                vis_row =  si/(pixel_width * visual_nr_of_hid_node);
                index_ptr_pix2hid_w = zero_ptr_pix2hid_w + si;
                ///Map over frome the large weight vevtor to visualize Mat
                ///============== This make so each patch row have a jump by 5 frames in the game ===
                show_ever5frame_start = (si/(pixel_height * visual_nr_of_hid_node * pixel_width)) * ((gameObj1.nr_of_frames-1) / visual_nr_of_frames);
                ///=====================================================================
                (*index_ptr_pix2hid_w) = pix2hid_weightB[(vis_row + show_ever5frame_start*pixel_height) * (pixel_width * Nr_of_hidden_nodes) + vis_colum%(pixel_width * Nr_of_hidden_nodes)];///Map over phuu...
                (*index_ptr_pix2hid_w) += 0.5f;///make gray scale center at 0.5f
            }
            imshow("pix2hid_weight", pix2hid_weight);///Only few weights showed
            imshow("hid2out_weight", hid2out_weight);
        }
        else
        {
            show_w_counter++;
        }
        ///========== End Show visualization of the weights =================
        printf("nr_of_episodes = %d\n", nr_of_episodes);
        if((nr_of_episodes% dice_dec_stddev_after_nr_episodes) == 0)
        {
            dice_stddev -= dice_stddev_decrease;
            if (dice_stddev < dice_minimum_stddev)
            {
                dice_stddev = dice_minimum_stddev;
            }
            gaussian_dice(1,0,true);
            printGaussianDiceSettings();
            
        }
        gameObj1.episode = nr_of_episodes;
        nr_of_episodes++;
        float rewards =0.0f;

     //   float pseudo_target =0.0f;///
        int ball_pad_diff = 0;/// Only used in high_precition_mode, Propotion reward mode
        ball_pad_diff = gameObj1.pad_ball_diff;
        ball_pad_diff = int_abs_value(ball_pad_diff);///remove sign only positive
        float win_flt_temp = 0.0f;
        if(gameObj1.win_this_game == 1)
        {            
            if(gameObj1.high_precition_mode==0)
            {
                rewards = +4.0f;///Yea.. the Ageint win this episode
            }
            else
            {
                ///Propotion reward mode
                printf("ball_pad_diff abs = %d\n", ball_pad_diff);
                if(ball_pad_diff == 0)
                {

                    ///Perfect catch
                    rewards = +20.0f;///Yea.. the Ageint win this episode
                }
                else
                {
                    rewards = 20.0f / (float) ball_pad_diff;
                }
                if(rewards > gameObj1.max_rewards)
                {
                    rewards = gameObj1.max_rewards;
                }
            }
        }
        else
        {            
            if(gameObj1.high_precition_mode==0)
            {
                rewards = -1.0f;///We lose this episode
            }
            else
            {
                ///Propotion reward mode
                printf("ball_pad_diff abs = %d\n", ball_pad_diff);
                if(ball_pad_diff == 0)
                {
                    ///Perfect catch WILL NOT happend when lose. just 0 diviton security
                }
                else
                {
                    rewards = - (((float) ball_pad_diff) / 60.0f ) ;
                }
                if(rewards < -3.0f)
                {
                    rewards = -3.0f;///Limit the negative rewards
                }
            }
        }
        
        if(rewards > 0.0f){ win_flt_temp = 1.0f; }
        else{ win_flt_temp = 0.0f; }

        if(win_lose_cnt < check_win_prob_ittr-1)
        {
            win_lose_cnt++;
            win_lose_sum = win_lose_sum + win_flt_temp;
        }
        else
        {
            win_probability = win_lose_sum / check_win_prob_ittr;
            win_lose_sum = 0.0f;
            win_lose_cnt=0;
        }
        if(win_lose_cnt>0){win_probability = win_lose_sum / win_lose_cnt;}
        printf("Win probablity = %f (ittr cnt = %d)\n", win_probability, win_lose_cnt);

        if(gameObj1.flip_reward_sign == 1)
        {
            printf("Flip sign of the rewards \n");
            rewards *= -1.0;
            ///Testing what happen with flipped reward
        }
        printf("rewards = %f\n", rewards);

        if(rewards > highest_rewards)
        {
            highest_rewards = rewards;
            best_rewards_serie = gameObj1.replay_count;
            ///Save this action serie for a later replay
            for(int i=0;i<gameObj1.nr_of_frames;i++)
            {
                best_actions[i] = action[i];
            }
            printf("best_rewards_serie = %d\n", best_rewards_serie);
        }
        if(gameObj1.replay_count == gameObj1.replay_times)
        {
            if(gameObj1.replay_times > 0)
            {
                gameObj1.replay_episode = 1;///Replay mode
                ///-------------------------- Replay best serie so with best actions it will contain ----------------
                gameObj1.start_episode();///Staring a new game turn
                randomize_dropoutHid(&dropoutHidden[0], (gameObj1.nr_of_frames * Nr_of_hidden_nodes), gameObj1.Not_dropout, gameObj1.drop_out_percent);///select dropout node to the hidden node
                for(int frame_g=0; frame_g<gameObj1.nr_of_frames; frame_g++) ///Loop throue each of the 100 frames
                {
                    output_node[frame_g] = 0.0f;///Start with clear this node
                    gameObj1.frame = frame_g;
                    gameObj1.run_episode();
                    test = gameObj1.gameGrapics.clone();
                    resize(test, resized_grapics, size);
                    if(gameObj1.use_image_diff==1)
                    {
                        if(frame_g==0)
                        {
                            pre_grap = Scalar(0.0f);
                        }
                        else
                        {
                            diff_grap = pre_grap - resized_grapics;
                            pre_grap = resized_grapics.clone();///Used to calculate diff_grap For next frame
                        }
                        //diff_grap = diff_grap + 0.5f;
                    }
                    ///=============== Forward data for this frame ==================
                    ///Make the Dot product to this frames hidden nodes and output node
                    for(int i=0; i<Nr_of_hidden_nodes; i++)
                    {
                        hidden_node[frame_g * Nr_of_hidden_nodes + i] = 0.0f;///Start with clear this value before sum up the dot product
                        dot_product = 0.0f;///Start with clear this value before sum up the dot product
                        for(int j=0; j<(pixel_height * pixel_width); j++)
                        {
                            ix = ((pixel_width * Nr_of_hidden_nodes) * (pixel_height * frame_g + j/pixel_width) + (pixel_width * i) + j%pixel_width);///Prepare the index to point on the right place in the weight matrix pix2hid_weightB[]
                            if(gameObj1.use_image_diff==1)
                            {
                                index_ptr_diff_grap = zero_ptr_diff_grap + j;///Prepare the pointer address to point on the right place on the grapical image of this grapical frame
                                dot_product += pix2hid_weightB[ix] * (*index_ptr_diff_grap);///Make the dot product of Weights * Game grapichs
                                input_node_stored[pixel_width * pixel_height * frame_g + j] = (*index_ptr_diff_grap);///Save this grame grapich pixel must read this pixel later when update weights
                            }
                            else
                            {
                                index_ptr_res_grap = zero_ptr_res_grap + j;///Prepare the pointer address to point on the right place on the grapical image of this grapical frame
                                dot_product += pix2hid_weightB[ix] * (*index_ptr_res_grap);///Make the dot product of Weights * Game grapichs
                                if(use_noise_image==1 && i==0)
                                {
                                    float noise;
                                    noise = ((float) (rand() % 65535) / 65536) ;
                                    if(noise < noise_thres)
                                    {
                                        (*index_ptr_res_grap) = ((float) (rand() % 65535) / 65536);
                                    }
                                }
                                input_node_stored[pixel_width * pixel_height * frame_g + j] = (*index_ptr_res_grap);///Save this grame grapich pixel must read this pixel later when update weights
                            }
                        }
                        ///Relu this dot product
                        if(dropoutHidden[(frame_g * Nr_of_hidden_nodes + i)] == 0)
                        {
                            ///Normal forward not drop out this node
                            dot_product = relu(dot_product);
                        }
                        else
                        {
                            dot_product = 0.0f;
                        }
                        hidden_node[frame_g * Nr_of_hidden_nodes + i] = dot_product;///Put this finnish data in the hidden node neuron
                        ix = (frame_g * Nr_of_hidden_nodes + i);
                        output_node[frame_g] += hid2out_weightB[ix] * dot_product;///Take this hidden node data how is for the moment => hidden_node[frame_g * Nr_of_hidden_nodes + i] = dot_product;
                    }
                    output_node[frame_g] = 1.0/(1.0 + exp(-(output_node[frame_g])));///Sigmoid function.  x = 1.0/(1.0 + exp(-(x)))
                    ///=============== End Forward data for this frame ==================
                    imshow("resized_grapics", resized_grapics);///  resize(src, dst, size);
                    if(gameObj1.use_image_diff==1)
                    {
                        imshow("diff", diff_grap);
                    }
                    waitKey(1);
                    ///=============== Made the action from stored best action serie =======
                    if(best_actions[frame_g] == 0.0f)
                    {
                        action[frame_g] = 0.0f;
                        gameObj1.move_up = 0;
                    }
                    if(best_actions[frame_g] == 1.0f)
                    {
                        action[frame_g] = 1.0f;
                        gameObj1.move_up = 1;
                    }
                    if(best_actions[frame_g] == 0.5f)
                    {
                        action[frame_g] = 0.5f;
                        gameObj1.move_up = 2;
                    }
                    ///================= End Action ==================
                }
                ///---------------------------End Replay best serie -------------------------------------------------
                printf("Best series was replayed\n");
                gameObj1.replay_episode = 0;
                rewards = highest_rewards;
                printf("replayed rewards = %f\n", rewards);
            }
            highest_rewards = -1000.0f;///clear to next series of run
        }

        if(gameObj1.replay_count == gameObj1.replay_times)
        {
            ///================== Make the backprop now when the hole episode is done ==============
         ///   for(int frame_g=0; frame_g<gameObj1.nr_of_frames; frame_g++) ///This loop thoue will only go thorue here to make backpropagate (not play game in the gameObj1)
            for(int frame_g = gameObj1.nr_of_frames; frame_g>0; true) ///This loop thoue will only go thorue here to make backpropagate (not play game in the gameObj1)
            {
                frame_g--;
                /// ***** Make a pseudo target value = action[frame_g] ******
                ///This is the cool stuff about Reinforcment Learning
                ///You pruduce a pseudo target value = action[frame_g] because then you can imidiet generate a gradient decent for this frame
                ///even if you don't know yet if this pseudo target is the right (in this case have right polarity +/- only because actions is only 2 = UP/DOWN )
                ///When the episode is over you can fix this eventually wrong +/- by the rewards
                output_delta[frame_g] = (action[frame_g] - output_node[frame_g]) * output_node[frame_g] * (1.0f - output_node[frame_g]);///Backpropagate. Make a gradient decent for this frame even if it may have wrong polarity
                for(int i=0; i<Nr_of_hidden_nodes; i++)
                {
                    ///**** Backprop delta_hid ****
                    ///delta_hid = delta_out * output_weight[1];
                    ix = (frame_g * Nr_of_hidden_nodes + i);

                    if(dropoutHidden[frame_g * Nr_of_hidden_nodes + i] == 0)
                        ///Correct bug 2018-05-03, now correct derivitiv for proper gradient decent at ReLU
                        float derivat_of_node = 0.0f;
                        if(hidden_node[frame_g * Nr_of_hidden_nodes + i] < 0.0f)
                        {
                            derivat_of_node = Relu_neg_gain;///ReLU-, derivate = Relu_neg_gain
                        else
                        {
                            derivat_of_node = 1.0;///ReLU+, derivate = 1
                        }
                        hidden_delta[frame_g * Nr_of_hidden_nodes + i] = output_delta[frame_g] * hid2out_weightB[ix] * derivat_of_node;///Relu Backprop to hidden delta

                    }
                    else
                    {
                        hidden_delta[frame_g * Nr_of_hidden_nodes + i]  = 0.0f;/// Hidden node delta zero when drop out no change of the weight regarding this backprop
                    }
                    ///===== Update weights depend on the stored delta ========

                    for(int j=0; j<(pixel_height * pixel_width); j++)
                    {
                        ix = ((pixel_width * Nr_of_hidden_nodes) * (pixel_height * frame_g + j/pixel_width) + (pixel_width * i) + j%pixel_width);
                        pix2hid_weightB[ix] += hidden_delta[frame_g * Nr_of_hidden_nodes + i] * input_node_stored[pixel_width * pixel_height * frame_g + j] * pix2hid_learning_rate * rewards;///    input_node_stored = new float[pixel_width * pixel_height * gameObj1.nr_of_frames];
                    }
                    ix = (frame_g * Nr_of_hidden_nodes + i);
                    hid2out_weightB[ix] += output_delta[frame_g] * hidden_node[frame_g * Nr_of_hidden_nodes + i] * hid2out_learning_rate * rewards;///Update weights
                    ///=============End update weights for this position ==============
                }
                ///printf("Rewards %f\n", rewards);
                rewards *= gamma;

            }
            ///=================== End Backprop ====================================================
        waitKey(1);
        }
    }
    return 0;
}


#ifndef FASTCCD_H
#define FASTCCD_H

#include "util.h"

class FastCCD
{
public:
    FastCCD();

    void parseParameter(int argc, char **argv);
    void dumpParameter();


    void updateW();
    void updateH();
    double updateLatent(int t, long i, long j, bool trans);

    void updateRt(int t, const vec_t &w);
    void updateR(int t, const vec_t &h);

    double calObj();

    void run();


private:
     void exit_with_help();
     void init();

public:
    int k;
    int threads;
    int maxiter, maxinneriter;
    double lambda;
    double rho;
    double eta0, betaup, betadown;  // learning rate parameters used in DSGD
    int lrate_method, num_blocks;
    int do_predict, verbose;
    bool with_weights;

    smat_t R, Rt;
    mat_t W,H;

    char input_file_name[1024];
    char model_file_name[1024];

    double loss;
    bool done;


};

#endif // FASTCCD_H

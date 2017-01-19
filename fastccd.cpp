#include "fastccd.h"
#include <omp.h>
#include <pthread.h>
#include <unistd.h>
#define kind dynamic,500

FastCCD::FastCCD()
{
    k = 10;
    rho = 1e-3;
    maxiter = 5;
    maxinneriter = 5;
    lambda = 0.1;
    threads = 4;
    eta0 = 1e-3; // initial eta0
    betaup = 1.05;
    betadown = 0.5;
    num_blocks = 30;  // number of blocks used in dsgd
    do_predict = 0;
    verbose = 0;


    with_weights = false;
}




void FastCCD::parseParameter(int argc, char **argv)
{

    int i;

    // parse options
    for(i=1;i<argc;i++)
    {
        if(argv[i][0] != '-') break;
        if(++i>=argc)
            exit_with_help();
        switch(argv[i-1][1])
        {

            case 'k':
                k = atoi(argv[i]);
                break;

            case 'n':
                threads = atoi(argv[i]);
                break;

            case 'l':
                lambda = atof(argv[i]);
                break;

            case 'r':
                rho = atof(argv[i]);
                break;

            case 't':
                maxiter = atoi(argv[i]);
                break;

            case 'T':
                maxinneriter = atoi(argv[i]);
                break;


            case 'B':
                num_blocks = atoi(argv[i]);
                break;

            case 'm':
                lrate_method = atoi(argv[i]);
                break;

            case 'u':
                betaup = atof(argv[i]);
                break;

            case 'd':
                betadown = atof(argv[i]);
                break;

            case 'p':
                do_predict = atoi(argv[i]);
                break;

            case 'q':
                verbose = atoi(argv[i]);
                break;


            default:
                fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
                exit_with_help();
                break;
        }
    }

    if (do_predict!=0)
        verbose = 1;

    // determine filenames
    if(i>=argc)
        exit_with_help();

    strcpy(input_file_name, argv[i]);

    if(i<argc-1)
        strcpy(model_file_name,argv[i+1]);
    else
    {
        char *p = argv[i]+ strlen(argv[i])-1;
        while (*p == '/')
            *p-- = 0;
        p = strrchr(argv[i],'/');
        if(p==NULL)
            p = argv[i];
        else
            ++p;
        sprintf(model_file_name,"%s.model",p);
    }

    omp_set_num_threads(threads);

    init();

    done = false;

}

void FastCCD::dumpParameter()
{
    cout<<"The paramters are:"<<endl;
    cout<<"k\t"<<k<<endl;
    cout<<"rho\t"<<rho<<endl;
    cout<<"maxiter\t"<<maxiter<<endl;
    cout<<"maxinneriter\t"<<maxinneriter<<endl;
    cout<<"lambda\t"<<lambda<<endl;
    cout<<"threads\t"<<threads<<endl;
    cout<<"eta0\t"<<eta0<<endl;
    cout<<"betaup\t"<<betaup<<endl;
    cout<<"betadown\t"<<betadown<<endl;
    cout<<"num_blocks\t"<<num_blocks<<endl;
    cout<<"do_predict\t"<<do_predict<<endl;
    cout<<"verbose\t"<<verbose<<endl;
    cout<<"input_file\t"<<input_file_name<<endl;
    cout<<"output_file\t"<<model_file_name<<endl;

}

void FastCCD::exit_with_help()
{
    printf(
    "Usage: omp-pmf-train [options] data_dir [model_filename]\n"
    "options:\n"
    "    -s type : set type of solver (default 0)\n"
    "    	 0 -- CCDR1 with fundec stopping condition\n"
    "    -k rank : set the rank (default 10)\n"
    "    -n threads : set the number of threads (default 4)\n"
    "    -l lambda : set the regularization parameter lambda (default 0.1)\n"
    "    -t max_iter: set the number of iterations (default 5)\n"
    "    -T max_iter: set the number of inner iterations used in CCDR1 (default 5)\n"
    "    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n"
    "    -p do_predict: do prediction or not (default 0)\n"
    "    -q verbose: show information or not (default 0)\n"
    "    -N do_nmf: do nmf (default 0)\n"
    );
}

void FastCCD::init()
{
    FILE *model_fp = NULL;

    if(model_file_name) {
        model_fp = fopen(model_file_name, "wb");
        if(model_fp == NULL)
        {
            fprintf(stderr,"can't open output file %s\n",model_file_name);
        }
    }

    testset_t T;

    load(input_file_name,R,T, with_weights);
    Rt = R.transpose();

    // W, H  here are m*k, n*k
    initial(W, R.rows, k);
    initial(H, R.cols, k);

    puts("starts!");


    if(model_fp) {
        save_mat_t(W,model_fp,false);
        save_mat_t(H,model_fp,false);
        fclose(model_fp);
    }

}



void FastCCD::updateW()
{

    for(int t=0; t<k; t++){

        vec_t w;
        w.reserve(Rt.cols);

 #pragma omp parallel for schedule(kind) shared(w)
        for(long i=0; i<Rt.cols; i++){
            w[i] = updateLatent(t, i, 0, true);
        }

        updateRt(t, w);


 #pragma omp parallel for
        for(long i=0; i<Rt.cols; i++){
            W[i][t] = w[i];
        }


    }
}

void FastCCD::updateH()
{
    for(int t=0; t<k; t++){

        vec_t h;
        h.reserve(R.cols);

 #pragma omp parallel for schedule(kind) shared(h)
        for(long j=0; j<R.cols; j++){
            h[j] = updateLatent(t, 0, j, false);
        }

        updateR(t, h);


 #pragma omp parallel for
        for(long j=0; j<R.cols; j++){
            H[j][t] = h[j];
        }

    }
}

double FastCCD::updateLatent(int t, long i, long j, bool trans)
{
    double g = 0, f = lambda;//*(R.col_ptr[j+1] - R.col_ptr[j]);

    if(trans){
        for(long idx = Rt.col_ptr[i]; idx<Rt.col_ptr[i+1]; idx++){
            int j = Rt.row_idx[idx];
            g += (Rt.val[idx] + W[i][t]*H[j][t])*H[j][t];
            f += H[j][t] * H[j][t];
        }
    }
    else{
        for(long idx = R.col_ptr[j]; idx<R.col_ptr[j+1]; idx++){
            int i = R.row_idx[idx];
            g += (R.val[idx] +  W[i][t]*H[j][t])*W[i][t];
            f += W[i][t] * W[i][t];
        }
    }

    return g/f;

}


double FastCCD::calObj()
{
    double loss = 0;

#pragma omp parallel for schedule(kind) reduction(+:loss)
    for(long j=0; j<R.cols; j++){
        double innerloss = 0;
        for(long idx =R.col_ptr[j]; idx<R.col_ptr[j+1]; idx++){
            innerloss += R.val[idx]*R.val[idx];
        }
        loss += innerloss;
    }

#pragma omp parallel for schedule(kind) reduction(+:loss)
    for(long r=0; r<R.rows; r++){
        double innerloss = 0;
        for(int i=0; i<k; i++){
           innerloss += W[r][i]*W[r][i]; //Rt.nnz_of_col(r)*
        }
        loss += innerloss;
    }

#pragma omp parallel for schedule(kind) reduction(+:loss)
    for(long c=0; c<R.cols; c++){
        double innerloss = 0;
        for(int i=0; i<k; i++){
            innerloss += H[c][i]*H[c][i]; //R.nnz_of_col(c)*
        }
        loss += innerloss;
    }

    return loss;

}




void FastCCD::run()
{
    for(int it = 0; it < 500; it++)
    {
        double start = omp_get_wtime();
        updateW();
        updateH();
        double obj = calObj();
        double end = omp_get_wtime();

        cout<<it<<": "<<obj<<",\ttime:"<<end-start<<"s"<<endl;
    }




}



void FastCCD::updateR(int t, const vec_t &h)
{
    double obj = 0;

#pragma omp parallel for schedule(kind) reduction(+:obj)
    for(long c=0; c<R.cols; c++){
        double ht = h[c];
        double innerloss = 0;
        for(long idx=R.col_ptr[c]; idx<R.col_ptr[c+1]; idx++){
            int i = R.row_idx[idx];
            R.val[idx] = R.val[idx] - (ht - H[c][t])*W[i][t];
            innerloss += R.val[idx]*R.val[idx];
        }
        obj += innerloss;
    }
    loss = obj;

#pragma omp parallel for schedule(kind)
    for(long c=0; c<Rt.cols; c++){
        for(long idx = Rt.col_ptr[c]; idx<Rt.col_ptr[c+1]; idx++){
            int i = Rt.row_idx[idx];
            Rt.val[idx] = Rt.val[idx] - ( h[i] - H[i][t] )*W[c][t];
        }

    }

}


void FastCCD::updateRt(int t, const vec_t &w)
{

 #pragma omp parallel for schedule(kind)
    for(long c=0; c<Rt.cols; c++){
        double wt = w[c];
        for(long idx = Rt.col_ptr[c]; idx<Rt.col_ptr[c+1]; idx++){
            int i = Rt.row_idx[idx];
            Rt.val[idx] = Rt.val[idx] - (wt- W[c][t] )*H[i][t];
        }

    }

 #pragma omp parallel for schedule(kind)
    for(long c=0; c<R.cols; c++){
        for(long idx=R.col_ptr[c]; idx<R.col_ptr[c+1]; idx++){
            int i = R.row_idx[idx];
            R.val[idx] = R.val[idx] - ( w[i] - W[i][t])*H[c][t];
        }
    }
}


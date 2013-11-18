#include "spmv.hpp"

int main(int argc, char* argv[]){
    if (argc < 2)
    {
        cout<<"Usage: ./spmv sparse.mtx [optimal.cfg]"<<endl;
        return 0;
    }
    char* filename = argv[1];
    FILE* infile = fopen(filename, "r");
    clContext clCxt;
    getClContext(&clCxt);
    timeRcd.min_totaltime=10000000;
    Plan best;
    CLBCCOO clbccoo;
    MTX<float> mtx;
    fileToMtx<float>(filename,&mtx);
    cl_mem vec_dev,res_dev;
    float *vec = new float[mtx.cols];
    for(int i=0;i<mtx.cols;i++) vec[i]=i;
    int tune = 1; 
    if(argc == 3){
        FILE* infile_1 = fopen(argv[2], "r");
        fscanf(infile_1, "%d", &best.localthread);
        fscanf(infile_1, "%d", &best.cta);
        fscanf(infile_1, "%d", &best.col_delta);
        fscanf(infile_1, "%d", &best.registergroup);
        fscanf(infile_1, "%d", &best.localmemgroup);
        fscanf(infile_1, "%d", &best.block_width);
        fscanf(infile_1, "%d", &best.block_height);
        fscanf(infile_1, "%d", &best.trans);
        fscanf(infile_1, "%d", &best.tx);
        fscanf(infile_1, "%d", &best.coalesced);
        fscanf(infile_1, "%d", &best.slices);
        fscanf(infile_1, "%d", &best.bitwidth);
        fscanf(infile_1, "%d", &best.dimwidth);
        fclose(infile_1);
        tune = 0;
    }
    //else
    //    generateProgramCache<float>(&clCxt,&mtx);
    
    yaSpMVmtx2clbccoo<float>(&clCxt,&mtx,&clbccoo,&best,tune);
    
    create(&clCxt,&res_dev,clbccoo.rows*sizeof(float));
    if(best.tx == 1)
        getTexture(&clCxt, vec_dev, vec, clbccoo.cols*sizeof(float), best.block_width,TEXTURE_WIDTH);
    else{
        create(&clCxt,&vec_dev,clbccoo.cols*sizeof(float));
        upload(&clCxt,(void *)vec,vec_dev,clbccoo.cols*sizeof(float));
    }
    cl_ulong cstart,cend;
    struct timeval vstart,vend;
    gettimeofday(&vstart,NULL);
    int run_times=1000;
    yaSpMVRun<float>(&clCxt, &clbccoo,vec_dev,res_dev,&best,run_times);
    clFinish(clCxt.command_queue);
    gettimeofday(&vend,NULL);
    cstart=(cl_ulong)vstart.tv_sec*1000000 + (cl_ulong)vstart.tv_usec;
    cend=(cl_ulong)vend.tv_sec*1000000 + (cl_ulong)vend.tv_usec;
    double t = (double)(cend -cstart)/1000/run_times;
    cout<<"Best Plan: lt="<<best.localthread<<" ca="<<best.cta<<" cd="<<best.col_delta;
    cout<<" wg="<<best.workgroup<<" rg="<<best.registergroup<<" lg="<<best.localmemgroup;
    cout<<" bl="<<best.block_width<<" bh="<<best.block_height<<" tr="<<best.trans;
    cout<<" tx="<<best.tx<<" co="<<best.coalesced<<" sl="<<best.slices;
    cout<<" bi="<<best.bitwidth<<" dw="<<best.dimwidth<<endl;
    
    cout<<"Execute time:"<<t<<" milliseconds."<<endl;
    return 0;
}

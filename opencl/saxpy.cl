__kernel void saxpy(__constant float* x, __global float* y, float a)
{
    const int i = get_global_id(0);

    y[i] = a * x[i] + y[i];
}

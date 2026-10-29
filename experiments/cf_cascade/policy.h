#pragma once
#include "risk_model.h"
namespace diskann {
struct Cascade {
 const CounterfactualModel* model=nullptr;
 int kind=0;
 float secondary=0;
 unsigned first_skip=0,second_calls=0,second_skip=0,misses=0;
};
inline thread_local Cascade* cascade=nullptr;
inline void cascade_apply(const float* state,float& score,float first_threshold) {
 if(!cascade || state[10]!=0)return;
 auto& c=*cascade;++c.misses;
 if(score<first_threshold){++c.first_skip;return;}
 if(c.kind==0)return;
 ++c.second_calls;
 bool skip=c.kind==1?c.model->predict(state)[0]<c.secondary:score<c.secondary;
 if(skip){++c.second_skip;score=-1.f;}
}
}

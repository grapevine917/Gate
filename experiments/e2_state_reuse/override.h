#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace diskann {
struct E2Override {
 uint32_t step=0,node=0,skip=0,seen=0,applied=0;
 float expected_score=0,state[17]{};
 float original_score=0;
};
inline thread_local E2Override* e2_override=nullptr;
inline void e2_override_score(uint32_t node,const float*state,float&score) {
 if(!e2_override)return;
 auto& h=*e2_override;
 if(h.seen++!=h.step)return;
 if(node!=h.node || state[10]!=0 || std::memcmp(state,h.state,17*sizeof(float)) || std::abs(score-h.expected_score)>2e-6)
  throw std::runtime_error("E2 intervention prefix state/node/score mismatch");
 h.original_score=score;score=h.skip?0.0f:1.0f;++h.applied;
}
}

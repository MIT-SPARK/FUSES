#include "graph.h"

#ifdef _MSC_VER
#pragma warning(disable: 4661)
#endif

// Instantiations: <captype, tcaptype, flowtype>
// IMPORTANT: 
//    flowtype should be 'larger' than tcaptype 
//    tcaptype should be 'larger' than captype

template class maxflowLib::Graph<int,int,int>;
template class maxflowLib::Graph<size_t,size_t,size_t>;
template class maxflowLib::Graph<short,int,int>;
template class maxflowLib::Graph<float,float,float>;
template class maxflowLib::Graph<double,double,double>;


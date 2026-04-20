#ifndef __FFTM_DETAIL_ARRAY_ARRANGERS_H__
#define __FFTM_DETAIL_ARRAY_ARRANGERS_H__

#include <contrib/scfd/test/arrays/custom_index_fast_arranger.h>

namespace scfd
{
namespace arrays
{

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_012_t = scfd::arrays::custom_index_fast_arranger<0, 1, 2>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_021_t = scfd::arrays::custom_index_fast_arranger<0, 2, 1>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_10_t = scfd::arrays::custom_index_fast_arranger<1, 0>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_102_t = scfd::arrays::custom_index_fast_arranger<1, 0, 2>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_120_t = scfd::arrays::custom_index_fast_arranger<1, 2, 0>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_201_t = scfd::arrays::custom_index_fast_arranger<2, 0, 1>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_210_t = scfd::arrays::custom_index_fast_arranger<2, 1, 0>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_0123_t = scfd::arrays::custom_index_fast_arranger<0, 1, 2, 3>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_0213_t = scfd::arrays::custom_index_fast_arranger<0, 2, 1, 3>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_3012_t = scfd::arrays::custom_index_fast_arranger<3, 0, 1, 2>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_1023_t = scfd::arrays::custom_index_fast_arranger<1, 0, 2, 3>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_3120_t = scfd::arrays::custom_index_fast_arranger<3, 1, 2, 0>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_3210_t = scfd::arrays::custom_index_fast_arranger<3, 2, 1, 0>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_1032_t = scfd::arrays::custom_index_fast_arranger<1, 0, 3, 2>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_2103_t = scfd::arrays::custom_index_fast_arranger<2, 1, 0, 3>::type<Dims...>;

} // namespace arrays
} // namespace scfd

#endif // __FFTM_DETAIL_ARRAY_ARRANGERS_H__

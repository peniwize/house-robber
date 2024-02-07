/*!
    \file "main.cpp"

    Author: Matt Ervin <matt@impsoftware.org>
    Formatting: 4 spaces/tab (spaces only; no tabs), 120 columns.
    Doc-tool: Doxygen (http://www.doxygen.com/)

    https://leetcode.com/problems/house-robber/
*/

//!\sa https://github.com/doctest/doctest/blob/master/doc/markdown/main.md
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "utils.hpp"

/*
    You are a professional robber planning to rob houses along a street.
    Each house has a certain amount of money stashed, the only constraint 
    stopping you from robbing each of them is that adjacent houses have 
    security systems connected and it will automatically contact the police 
    if two adjacent houses were broken into on the same night.

    Given an integer array nums representing the amount of money of 
    each house, return the maximum amount of money you can rob tonight 
    without alerting the police.

    Constraints:

        * 1 <= nums.length <= 100
        * 0 <= nums[i] <= 400
*/

class Solution {
public:
    int rob(vector<int>& nums) {

//
//!\todo TODO: >>> Under Construction <<<
//
return -1;

    }
};

// {----------------(120 columns)---------------> Module Code Delimiter <---------------(120 columns)----------------}

namespace doctest {
    const char* testName() noexcept { return doctest::detail::g_cs->currentTest->m_name; }
} // namespace doctest {

TEST_CASE("Case 1")
{
    cerr << doctest::testName() << '\n';
    auto nums = vector<int>{
        1,2,3,1
    };
    auto const expected = 4;
    auto solution = Solution{};
    { // New scope.
        auto const start = std::chrono::steady_clock::now();
        auto const result = solution.rob(nums);
        CHECK(expected == result);
        cerr << "Elapsed time: " << elapsed_time_t{start} << '\n';
    }
    cerr << "\n";
}

TEST_CASE("Case 2")
{
    cerr << doctest::testName() << '\n';
    auto nums = vector<int>{
        2,7,9,3,1
    };
    auto const expected = 12;
    auto solution = Solution{};
    { // New scope.
        auto const start = std::chrono::steady_clock::now();
        auto const result = solution.rob(nums);
        CHECK(expected == result);
        cerr << "Elapsed time: " << elapsed_time_t{start} << '\n';
    }
    cerr << "\n";
}

/*
    End of "main.cpp"
*/

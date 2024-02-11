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

/*
    Simplified iterative DP solution (THREE variables).

    This is equivalent to the iterative solution that simulates the return path
    of the recursive solution, except that this solution works FORWARD instead 
    of backward and it eliminates superfluous tests for edge cases, which 
    simplifies the code.  This also runs faster because the iteration loop is
    as simple as possible (moving forward one value [not index] at a time),
    which executes much faster in python (and C++).

    See 'Solution4_OptimizedDP' (in "solution.py")
    for more explanation and context.
    https://leetcode.com/problems/house-robber/solutions/4710034/python-dp-two-vars-full-explanation-t-o-n-s-o-1/

    Time = O(n)

    Space = O(1)
*/
class Solution5_OptimizedDP {
public:
    int rob(vector<int>& nums) {
        int a = 0, b = 0, c = 0;

        for (auto const n : nums) {
            c = n + (std::max)(b, c);
            std::swap(a, c);
            std::swap(b, c);
        }

        return (std::max)(a, b);
    }
};

/*!
    Simplified iterative DP solution (TWO variables).

    See 'Solution5_OptimizedDP' for more explanation and context.

    An alternative way of looking at the problem is to only consider the 
    house to rob (at 'i') and the previous two: 'i - 1' and 'i - 2'.  All house
    combinations leading up to 'i - 1' and 'i - 2' will be reflected in the
    max values computed for those indexes.  Therefore only 'i - 1' and 'i - 2'
    must be considered (similar to the Fibonacci algorithm).  This reduces the
    number of variables from three to two.

    p1, p2 = 0, 0
    for n in nums:
        tmp = max(p2 + n, p1)
        p2 = p1
        p1 = tmp
    return p1
    
    'p1' is the previous [adjacent] house to 'n'.
    'p2' is the house before that (two houses away from 'n').
    The idea is that if you chose to rob 'n' then you would have also robed 'p2',
    because they're not adjacent, so you add the value of the haul from robbing
    house 'p2' to 'n'.  Whether or not to rob p1 or p2 depends on which robbery
    will produce the largest haul (decided by the max() function).  'p1' and 
    'p2' EACH INCLUDE THE MAXIMUM POSSIBLE HAUL FROM _ALL_ COMBINATIONS OF 
    ROBBERIES THAT LED TO EACH OF THEM.  Consider the following set of all 
    combinations that come from seven houses: {a, b, c, d, e, f, g}.
    Starting with the first house 'a':
    
       {a b c d e f g}
       ---------------
     0: ∅  (skipped by this algorithm; no sense in robbing no houses)
     1: a
     2: a   c
     3: a   c   e
     4: a   c   e   g

    The next combination removes 'e' and resumes with 'f', then 'g'.

     5: a   c     f
     6: a   c       g

    The next combination removes 'c' and resumes with 'd'.
    Note that the house combinations that start with 'a' include ALL 
    combinations, not just every other house.  This is why 'p1' and 'p2' 
    include the effects of the decision tree choices that led up to them.
    'p1' and 'p2' are each an amalgamation of all prior decisions/combinations.

     7: a     d
     8: a     d   f
     9: a     d     g
    10: a       e
    11: a       e   g
    12: a         f
    13: a           g
    14:   b
    15:   b   d
    16:   b   d   f
    17:   b   d     g
    18:   b     e
    19:   b     e   g
    20:   b       f
    21:   b         g
    22:     c
    23:     c   e
    24:     c   e   g
    25:     c     f
    26:     c       g
    27:       d
    28:       d    f
    29:       d      g
    30:         e
    31:         e    g
    32:           f
    33:              g
*/
class Solution6_OptimizedDP {
public:
    int rob(vector<int>& nums) {
        int p1 = 0, p2 = 0;

        for (auto const n : nums) {
            auto const tmp = (std::max)(n + p2, p1);
            p2 = p1;
            p1 = tmp;
        }

        return p1;
    }
};

// {----------------(120 columns)---------------> Module Code Delimiter <---------------(120 columns)----------------}

#define Solution Solution6_OptimizedDP

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

TEST_CASE("Case 3")
{
    cerr << doctest::testName() << '\n';
    auto nums = vector<int>{
        183,219,57,193,94,233,202,154,65,240,97,234,100,249,186,66,90,238,168,128,177,235,50,81,185,165,217,207,88,80,112,78,135,62,228,247,211
    };
    auto const expected = 3365;
    auto solution = Solution{};
    { // New scope.
        auto const start = std::chrono::steady_clock::now();
        auto const result = solution.rob(nums);
        CHECK(expected == result);
        cerr << "Elapsed time: " << elapsed_time_t{start} << '\n';
    }
    cerr << "\n";
}

TEST_CASE("Case 4")
{
    cerr << doctest::testName() << '\n';
    auto nums = vector<int>{
        1,2
    };
    auto const expected = 2;
    auto solution = Solution{};
    { // New scope.
        auto const start = std::chrono::steady_clock::now();
        auto const result = solution.rob(nums);
        CHECK(expected == result);
        cerr << "Elapsed time: " << elapsed_time_t{start} << '\n';
    }
    cerr << "\n";
}

TEST_CASE("Case 5")
{
    cerr << doctest::testName() << '\n';
    auto nums = vector<int>{
        2,1,1,2
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

TEST_CASE("Case 100")
{
    cerr << doctest::testName() << '\n';
    auto nums = vector<int>{
        1,3,21,1,0,1,1,0,64,1,0,1,1,7
    };
    auto const expected = 95;
    auto solution = Solution{};
    { // New scope.
        auto const start = std::chrono::steady_clock::now();
        auto const result = solution.rob(nums);
        CHECK(expected == result);
        cerr << "Elapsed time: " << elapsed_time_t{start} << '\n';
    }
    cerr << "\n";
}

TEST_CASE("Case 101")
{
    cerr << doctest::testName() << '\n';
    auto nums = vector<int>{
        15
    };
    auto const expected = 15;
    auto solution = Solution{};
    { // New scope.
        auto const start = std::chrono::steady_clock::now();
        auto const result = solution.rob(nums);
        CHECK(expected == result);
        cerr << "Elapsed time: " << elapsed_time_t{start} << '\n';
    }
    cerr << "\n";
}

TEST_CASE("Case 102")
{
    cerr << doctest::testName() << '\n';
    auto nums = vector<int>{
    };
    auto const expected = 0;
    auto solution = Solution{};
    { // New scope.
        auto const start = std::chrono::steady_clock::now();
        auto const result = solution.rob(nums);
        CHECK(expected == result);
        cerr << "Elapsed time: " << elapsed_time_t{start} << '\n';
    }
    cerr << "\n";
}

TEST_CASE("Case 103")
{
    cerr << doctest::testName() << '\n';
    auto nums = vector<int>{
        1,2,3
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

/*
    End of "main.cpp"
*/

# Start of "solution.py".

from collections import deque
import inspect
import time
from typing import List
from typing import Optional
from typing import Set

"""
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
"""

"""
    Brute force solution.

    Return the max sum of all combinations with non-adjacent elements:

    This problem is solved by calculating every combination of houses that
    does not have any adjacent houses.  For example, consider a street with
    seven houses: {a,b,c,d,e,f,g}.  The decision tree that produces all
    combinations of these houses is:

                         $
                         |
    +----------------------------------------------------+
    |                    |                               |
    ∅                    a                               b
                         |                               |
       +------+----------+-------+---+---+       +-------+---+---+
       |      |          |       |   |   |       |       |   |   |
       ∅      c          d       e   f   g       d       e   f   g
              |          |       |               |       |
          +--------+  +-----+  +---+          +-----+  +---+
          |  |  |  |  |  |  |  |   |          |  |  |  |   |
          ∅  e  f  g  ∅  f  g  ∅   g          ∅  f  g  ∅   g
             |
           +---+
           |   |
           ∅   g
    
    NOTE: '∅' denotes no choice or "nothing".

    Notice that sub-trees 'a' and 'b' both contain all other sub-trees and,
    similarly, all sub-trees are composites that contain all other sub-trees
    all the way to the base cases: 'f' and 'g'.
    The reason why only sub-trees 'a' and 'b' are modeled is because this 
    combination algorithm skips adjacent elements (houses) and both trees 
    must be modeled to show ALL nodes since neither tree 'a' nor tree 'b' 
    contains the other, however both contain all other sub-trees.

    Here are all combinations that are produced by the decision tree above:

    {a b c d e f g}
    ---------------
    ∅
    a
    a    c
    a    c   e
    a    c   e   g
    a    c     f
    a    c       g
    a      d
    a      d   f
    a      d     g
    a        e
    a        e   g
    a          f
    a            g
      b
      b    d
      b    d   f
      b    d     g
      b      e
      b      e   g
      b        f
      b          g
        c
        c    e
        c    e   g
        c      f
        c        g
          d
          d    f
          d      g
            e
            e    g
              f
                 g

    Note that every single combination can be found as a branch in the 
    decision tree and every branch in the decision tree has a corresponding
    combination.

    To solve this problem, the "max" value of each node must be calculated.
    The "max" value of a node is the node value + the max value of all of 
    the nodes children.  For example, the "max" value of any node is:
        node_max = node_value + max(node_max for all children)
    Which is a recursive function that returns the "max" value of a node to 
    its parent node that in turn adds its value to this "max" value, which 
    produces the "max" value of the parent node, which it returns to its 
    parent.  This continues until the root node is reached.  The answer to 
    the problem is the max value of the "max" values of 'a' and 'b'.

    The total number of PATHS to leaf nodes in a decision tree that DOES
    include adjacent houses is 2**n, where n is the number of choices 
    (2**7 = 128 in the example).  This is the number of unique combinations 
    (not permutations) that can be produced from the choices.

    This problem, however, does not produce ALL possible combinations because 
    adjacent houses (choices) are excluded.
    
    Time = Somewhere between O(2**(n/2)) and O(2**n).
            The algorithm is exponential and dependent on 'n'.  It will certainly run
            in less time than O(2**n) because a level of the decision tree is omitted
            by 'i + 2' passed to '__helper()' and it will certainly run in more time
            than O(2**(n/2)).  I'm not sure exactly how to calculate the precise base
            since '2' (in 2**n) is too large.

    Space = O(n/2) => O(n)  [call stack]
"""
class Solution1_BruteForce:
    def rob(self, nums: List[int]) -> int:
        def __helper(nums: List[int], startIdx: int = 0, sum: int = 0, maxSum: int = 0) -> int:
            if len(nums) <= startIdx:
                return max(maxSum, sum)
            for i in range(startIdx, len(nums)):
                sum += nums[i]
                maxSum = __helper(nums, i + 2, sum, maxSum)
                sum -= nums[i]
            return maxSum
        return __helper(nums)

"""
    Cached recursive [dynamic programming] solution.

    This is a recursive brute force solution that uses a cache to avoid
    redundantly reprocessing the same sub-tree more than once.  However, 
    the root of the sub-tree is still visited.  This solution takes advantage 
    of the fact the decision tree is composed of recurring identical 
    subproblems.  Each element in the cache corresponds to an index into nums,
    which represents a node in the decision tree.

    See 'Solution1_OptimizedDP' for more explanation and context.
    
    Time = O(n)
           Due to the cache cache (memoization), each node is only visited once,
           which prunes off all combination [sub]tree branches except one.

    Space = O(n + n/2) => O(n + n) => O(2n) => O(n)
            Term 1 (n): the cache cache, which may contain one value for each sub-tree.
            Term 2 (n/2): the maximum call stack depth.
"""
class Solution2_BruteForceDP:
    def rob(self, nums: List[int]) -> int:
        if not nums: return 0

        N = len(nums)        
        cache = {} # "nums" index to corresponding sub-tree node "max-value".

        def __nodeMaxSum(numsIdx: int = 0) -> int:
            if N <= numsIdx: return 0
            maxSum = 0
            for i in range(numsIdx, N):
                if i in cache:
                    sum = cache[i]
                else:
                    sum = nums[i] + __nodeMaxSum(i + 2)
                    cache[i] = sum
                maxSum = max(maxSum, sum)
            return maxSum
        
        return __nodeMaxSum()

""""
    Iterative [dynamic programming] solution [S=O(n)].

    This is an iterative (non-recursive) solution that takes advantage of 
    the fact the decision tree is composed of recurring identical subproblems.
    Each element in the cache is calculated as:
        cache[i] = nums[i] + max(cache[i+2], cache[i+3])
    The last two values (n - 1 and n - 2) in the cache are initialized to 
    the last two values in nums because the last two values in nums are 
    calculated as:
        cache[n - 1] = nums[n - 1] + max(0, 0)
        cache[n - 2] = nums[n - 2] + max(0, 0)

    NOTE: Each element in nums represents a node in the decision tree.

    Start with the last three elements (tree nodes) in nums and work your 
    way backward to the first element (tree node) in nums.  For each element
    (tree node) in nums, calculate the max-node-value of the node as the 
    sum of the node value and the maximum of all its child max-node-values.
    
    NOTE: the SUM of each node and its children is not what is being 
    calculated and stored in the cache.  Rather, it is the sum of the 
    node value and the max-node-value of the previously calculated 
    max-node-values of each of its children.

    See 'Solution2_OptimizedDP' for more explanation and context.

    Time = O(n - 2) => O(n)

    Space = O(n)
"""
class Solution3_OptimizedDP:
    def rob(self, nums: List[int]) -> int:
        if not nums: return 0
        
        N = len(nums)
        
        if 0 == N: return 0
        if 1 == N: return nums[0]
        if 2 == N: return max(nums[0], nums[1])

        # Work backward to simulate the return path of the recursive solution.
        # This can be refactored to work forward because the combinations can
        # be produced either from left to right or from right to left.
        cache = [0] * N # "max-node-values"
        cache[N - 1] = nums[N - 1]
        cache[N - 2] = nums[N - 2]
        cache[N - 3] = nums[N - 3] + nums[N - 1]
        for i in range(N - 4, -1, -1):
            cache[i] = nums[i] + max(cache[i + 2], cache[i + 3])

        return max(cache[0], cache[1])

"""
    Iterative [dynamic programming] solution [S=O(1)].

    This solution uses THREE variables to record/track max-node-values.
    This solution differs from the cached brute force solution, which visits 
    every leaf in the decision tree and calculates a unique sum for every 
    decision tree branch.  This solution takes advantage of the fact the 
    decision tree is composed of recurring identical subproblems and only the
    solutions for the subsequent THREE sub-problems are required to solve any
    sub-problem:
        cache[i] = nums[i] = (cache[i - 2], cache[i - 3])
        where: "cache[i]" contains the "max node value" for the [sub]tree node
               corresponding to "nums[i]".
    Rather than keeping a cache array with every previously calculated 
    max-node-value, only three varables are kept in order to track the three
    most recently computed max-node-values.  Each time a new value is 
    calculated, all previous values are shifted through the variables 
    (discarding the oldest) and the newly calculated value is stored in the 
    appropriate variable.  A three element array could be used instead of 
    three discrete variables, however it would probably be slower - 
    especially in python.

    See 'Solution3_OptimizedDP' for more explanation and context.

    Time = O(n)

    Space = O(1)
"""
class Solution4_OptimizedDP:
    def rob(self, A: List[int]) -> int:
        if not A: return 0
        
        N = len(A)
        
        if 0 == N: return 0
        if 1 == N: return A[0]
        if 2 == N: return max(A[0], A[1])

        # Work backward to simulate the return path of the recursive solution.
        # This can be refactored to work forward because the combinations can
        # be produced either from left to right or from right to left.
        c = A[N - 1]
        b = A[N - 2]
        a = A[N - 3] + c
        for i in range(N - 4, -1, -1):
            a, b, c = A[i] + max(b, c), a, b

        return max(a, b)

"""
    Simplified iterative [dynamic programming] solution.

    This is equivalent to the iterative solution that simulates the return path
    of the recursive solution, except that this solution works FORWARD instead 
    of backward and it eliminates superfluous tests for edge cases, which 
    simplifies the code.  This also runs faster because the iteration loop is
    as simple as possible (moving forward one value [not index] at a time),
    which executes much faster in python (and C++).

    See 'Solution4_OptimizedDP' for more explanation and context.

    Time = O(n)

    Space = O(1)
"""
class Solution5_OptimizedDP:
    def rob(self, A: List[int]) -> int:
        a, b, c = 0, 0, 0
        for n in A:
            a, b, c = n + max(b, c), a, b
        return max(a, b)

def test1(solution):
    nums = [1,2,3,1]
    expected = 4
    startTime = time.time()
    result = solution.rob(nums)
    endTime = time.time()
    print("{}:{}({:.6f} sec) result = {}".format(inspect.currentframe().f_code.co_name, type(solution), endTime - startTime, result))
    assert(expected == result)

def test2(solution):
    nums = [2,7,9,3,1]
    expected = 12
    startTime = time.time()
    result = solution.rob(nums)
    endTime = time.time()
    print("{}:{}({:.6f} sec) result = {}".format(inspect.currentframe().f_code.co_name, type(solution), endTime - startTime, result))
    assert(expected == result)

def test3(solution):
    nums = [183,219,57,193,94,233,202,154,65,240,97,234,100,249,186,66,90,238,168,128,177,235,50,81,185,165,217,207,88,80,112,78,135,62,228,247,211]
    expected = 3365 # Proved with brute force solution.
    startTime = time.time()
    result = solution.rob(nums)
    endTime = time.time()
    print("{}:{}({:.6f} sec) result = {}".format(inspect.currentframe().f_code.co_name, type(solution), endTime - startTime, result))
    assert(expected == result)

def test4(solution):
    nums = [1,2]
    expected = 2
    startTime = time.time()
    result = solution.rob(nums)
    endTime = time.time()
    print("{}:{}({:.6f} sec) result = {}".format(inspect.currentframe().f_code.co_name, type(solution), endTime - startTime, result))
    assert(expected == result)

def test5(solution):
    nums = [2,1,1,2]
    expected = 4
    startTime = time.time()
    result = solution.rob(nums)
    endTime = time.time()
    print("{}:{}({:.6f} sec) result = {}".format(inspect.currentframe().f_code.co_name, type(solution), endTime - startTime, result))
    assert(expected == result)

def test100(solution):
    nums = [1,3,21,1,0,1,1,0,64,1,0,1,1,7]
    expected = 95
    startTime = time.time()
    result = solution.rob(nums)
    endTime = time.time()
    print("{}:{}({:.6f} sec) result = {}".format(inspect.currentframe().f_code.co_name, type(solution), endTime - startTime, result))
    assert(expected == result)

def test101(solution):
    nums = [15]
    expected = 15
    startTime = time.time()
    result = solution.rob(nums)
    endTime = time.time()
    print("{}:{}({:.6f} sec) result = {}".format(inspect.currentframe().f_code.co_name, type(solution), endTime - startTime, result))
    assert(expected == result)

def test102(solution):
    nums = []
    expected = 0
    startTime = time.time()
    result = solution.rob(nums)
    endTime = time.time()
    print("{}:{}({:.6f} sec) result = {}".format(inspect.currentframe().f_code.co_name, type(solution), endTime - startTime, result))
    assert(expected == result)

def test103(solution):
    nums = [1,2,3]
    expected = 4
    startTime = time.time()
    result = solution.rob(nums)
    endTime = time.time()
    print("{}:{}({:.6f} sec) result = {}".format(inspect.currentframe().f_code.co_name, type(solution), endTime - startTime, result))
    assert(expected == result)

if "__main__" == __name__:
    test1(Solution1_BruteForce())
    test1(Solution2_BruteForceDP())
    test1(Solution3_OptimizedDP())
    test1(Solution4_OptimizedDP())
    test1(Solution5_OptimizedDP())

    test2(Solution1_BruteForce())
    test2(Solution2_BruteForceDP())
    test2(Solution3_OptimizedDP())
    test2(Solution4_OptimizedDP())
    test2(Solution5_OptimizedDP())

    #test3(Solution1_BruteForce())
    test3(Solution2_BruteForceDP())
    test3(Solution3_OptimizedDP())
    test3(Solution4_OptimizedDP())
    test3(Solution5_OptimizedDP())

    test4(Solution1_BruteForce())
    test4(Solution2_BruteForceDP())
    test4(Solution3_OptimizedDP())
    test4(Solution4_OptimizedDP())
    test4(Solution5_OptimizedDP())

    test5(Solution1_BruteForce())
    test5(Solution2_BruteForceDP())
    test5(Solution3_OptimizedDP())
    test5(Solution4_OptimizedDP())
    test5(Solution5_OptimizedDP())

    test100(Solution1_BruteForce())
    test100(Solution2_BruteForceDP())
    test100(Solution3_OptimizedDP())
    test100(Solution4_OptimizedDP())
    test100(Solution5_OptimizedDP())

    test101(Solution1_BruteForce())
    test101(Solution2_BruteForceDP())
    test101(Solution3_OptimizedDP())
    test101(Solution4_OptimizedDP())
    test101(Solution5_OptimizedDP())

    test102(Solution1_BruteForce())
    test102(Solution2_BruteForceDP())
    test102(Solution3_OptimizedDP())
    test102(Solution4_OptimizedDP())
    test102(Solution5_OptimizedDP())

    test103(Solution1_BruteForce())
    test103(Solution2_BruteForceDP())
    test103(Solution3_OptimizedDP())
    test103(Solution4_OptimizedDP())
    test103(Solution5_OptimizedDP())

# End of "solution.py".

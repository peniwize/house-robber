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
    Brute force solution: max sum of all combinations with non-adjacent elements.

    a b c d e f
    -----------
    a            = a
    a   c        = a+c
    a   c   e    = a+c+e
    a   c     f  = a+c+f
    a     d      = a+d
    a     d   f  = a+d+f
    a       e    = a+e
    a         f  = a+f

        b          = b
        b   d      = b+d
        b   d   f  = b+d+f
        b     e    = b+e
        b       f  = b+f

        c        = c
        c   e    = c+e
        c     f  = c+f

            d      = d
            d   f  = d+f

            e    = e

                f  = f

    Decision tree for all combinations in [five] choices {a, b, c, d, e}:
                                                    $
                            ∅                                               a
                ∅                       b                       ∅                       b
        ∅           c           ∅           c           ∅           c           ∅           c
        ∅     d     ∅     d     ∅     d     ∅     d     ∅     d     ∅     d     ∅     d     ∅     d
    ∅ e   ∅ e   ∅ e   ∅ e   ∅ e   ∅ e   ∅ e   ∅ e   ∅ e   ∅ e   ∅ e   ∅ e   ∅ e   ∅ e   ∅ e   ∅ e
    ↓ ↓   ↓ ↓   ↓ ↓   ↓ ↓   ↓ ↓   ↓ ↓   ↓ ↓   ↓ ↓   ↓ ↓   ↓ ↓   ↓ ↓   ↓ ↓   ↓ ↓   ↓ ↓   ↓ ↓   ↓ ↓
    ∅ e   d d   c c     c   b b   b b   b b   b b   a a   a a   a a   a a   a a   a a   a a   a a
            e     e     d     e   d d   c c   c c     e   d d   c c   c c   b b   b b   b b   b b
                        e           e     e   d d           e     e   d d     e   d d   c c   c c
                                                e                       e           e     e   d d
                                                                                                e

    The total number of PATHS to leaf nodes in the decision tree is 2**n, 
    where n is the number of choices, which is 2**5 = 32.  This is the number of 
    unique combinations (not permutations) that can be produced from the choices.

    This problem, however, does not produce ALL possible combinations because adjacent 
    houses (choices) cannot be considered.
    
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
    Dynamic programming solution.

    This is a recursive brute force solution that uses a cache to avoid
    redundantly reprocessing the same sub-tree more than once.  However, 
    the root of the sub-tree is still visited.

    Time = O(n)
           Due to the sums cache (memoization), each node is visited only once,
           which prunes off all combination tree branches except one.

    Space = O(n + n/2) => O(n + n) => O(2n) => O(n)
            Term 1 (n): the sums cache, which may contain one value for each sub-tree.
            Term 2 (n/2): the maximum call stack depth.
"""
class Solution2_BruteForceDP:
    def rob(self, nums: List[int]) -> int:
        if not nums: return 0

        N = len(nums)        
        sums = {}

        def __nodeMaxSum(numsIdx: int = 0) -> int:
            if N <= numsIdx: return 0
            maxSum = 0
            for i in range(numsIdx, N):
                if i in sums:
                    sum = sums[i]
                else:
                    sum = nums[i] + __nodeMaxSum(i + 2)
                    sums[i] = sum
                maxSum = max(maxSum, sum)
            return maxSum
        
        return __nodeMaxSum()

"""
    Dynamic programming solution.

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

        A = nums
        S = [0] * N
        S[N - 1] = A[N - 1]
        S[N - 2] = A[N - 2]
        S[N - 3] = A[N - 3] + A[N - 1]
        for i in range(N - 4, -1, -1):
            S[i] = A[i] + max(S[i + 2], S[i + 3])

        return max(S[0], S[1])

"""
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
    The reason why onlt sub-trees 'a' and 'b' are modeled is because this 
    combination algorithm skips adjacent elements (houses) and both trees 
    must be modeled to show ALL nodes since neither tree 'a' nor tree 'b' 
    contains the other.
    Here are all combinations that are produced by these decision trees:

    {a b c d e f g}
    ---------------
    ∅
    a
    a    c
    a    c   e
    a    c   e   g
    a    c     f
    a    c       g
    a      d   f
    a      d     g
    a        e
    a        e   g
    a          f
    a            g
    a        e
    a        e   g
    a          f
    a            g
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
        max_node = node_value + max(max_node for all children)
    Which is recursive a recursive function that returns the "max" value of
    a node to its parent node that adds it's value to this max value, which
    produces the max value of the parent node, which it returns to its parent.
    This continues until the root node is reached.  The answer to the problem
    is the max value of the max values of 'a' and 'b'.

    ((( Dynamic programming solution. )))

    This solution uses only three variables to record/track sums 
    rather than an array that records/tracks all sums.
    (This solution is similar to the Fibonacci DP solution.)

    Start with the last three elements and work your way backward to the 
    first element (in nums).  For each element in nums (each element 
    represents a node in the decision tree), calculate the MAX value of the 
    decision tree NODE as the sum of the node value and the max of all child 
    node MAX values.  Note that the SUM of each node and its children is not 
    what is being calculated!

    
>>> Under Construction <<<


    .  The sum values are calculated with the formula:
        S[i] = max(A[i] + S[i + 2], S[i + 1])
        where: A = nums array, S = sums array.
    The final result is S[0].

    This works because each sum depends on (includes) the subsequent sums,
    similar to how each Fibonacci number depends on the previous two numbers.
    The sums accumulate as the algorithm progresses.  Each sum in the sums 
    array depends only on the subsequent TWO, therefore a sums _array_ is 
    not needed, rather only TWO variables are required.  Let these variables
    be 'a' and 'b'.  Initialze them to the last two values in nums.  Then 
    calculate the next max value with:
        tmp = max(A[i] + a, b)
        b = a
        a = tmp
        where: A = nums array.
    The final result is in variable 'a'.

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

        c = A[N - 1]
        b = A[N - 2]
        a = A[N - 3] + c
        for i in range(N - 4, -1, -1):
            c = A[i] + max(b, c) # Use 'c' as tmp storage.
            a,b,c = c,a,b

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

    test2(Solution1_BruteForce())
    test2(Solution2_BruteForceDP())
    test2(Solution3_OptimizedDP())
    test2(Solution4_OptimizedDP())

    #test3(Solution1_BruteForce())
    test3(Solution2_BruteForceDP())
    test3(Solution3_OptimizedDP())
    test3(Solution4_OptimizedDP())

    test4(Solution1_BruteForce())
    test4(Solution2_BruteForceDP())
    test4(Solution3_OptimizedDP())
    test4(Solution4_OptimizedDP())

    test5(Solution1_BruteForce())
    test5(Solution2_BruteForceDP())
    test5(Solution3_OptimizedDP())
    test5(Solution4_OptimizedDP())

    test100(Solution1_BruteForce())
    test100(Solution2_BruteForceDP())
    test100(Solution3_OptimizedDP())
    test100(Solution4_OptimizedDP())

    test101(Solution1_BruteForce())
    test101(Solution2_BruteForceDP())
    test101(Solution3_OptimizedDP())
    test101(Solution4_OptimizedDP())

    test102(Solution1_BruteForce())
    test102(Solution2_BruteForceDP())
    test102(Solution3_OptimizedDP())
    test102(Solution4_OptimizedDP())

    test103(Solution1_BruteForce())
    test103(Solution2_BruteForceDP())
    test103(Solution3_OptimizedDP())
    test103(Solution4_OptimizedDP())

# End of "solution.py".

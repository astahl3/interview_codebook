'''
This file contains questions from: https://leetcode.com

Please refer to the website for the complete prompts

The code is organized as follows:
    
    ###########################################################################
    # LEETCODE PROBLEM: XYZ. Full Name of Problem [DIFFICULTY RATING]
    ###########################################################################
    
    # Solution to above problem goes here
    

Each solution is encapsulated in a docstring (i.e., surrounded by 3 quotes on 
at both ends). 

Only one solution should be active at a time. Other test cases can be added to
the main() function of each problem solution.

Solutions are typically O(N) whenever possible, where N is the length of the
input argument. However, some solutions may not be optimal as I personally
wrote them all.

Some code blocks contain * multiple * solutions. Be careful to adjust the
function call in main() appropriately to the desired answer in these cases.

'''

###############################################################################
# LEETCODE PROBLEM: 169. Majority Element [EASY]
###############################################################################
'''
class Solution:
    def majorityElement(self, nums):
        
        unique_nums_set= set(nums)
        unique_nums = list(unique_nums_set)
        L = len(nums)
        for k in range(0,len(unique_nums)):
            if nums.count(unique_nums[k]) > L/2:
                return unique_nums[k]
        return 0

def main():
   
    nums = [3,2,3,3,3,4,5,3,3]
    
    solution = Solution()
    result = solution.majorityElement(nums)
    print("Majority Element: ", result)
   
if __name__ == "__main__":
    main()
 '''  
   
   
###############################################################################
# LEETCODE PROBLEM: 55. Jump Game [MEDIUM]
###############################################################################    
'''
class Solution:  
    def canJump(self, nums):
       
        L = len(nums)
        nums2 = list(nums)
       
        zeros_removed = 0
        Lzeros = 1
        temp_arr = []
       
        while 0 in nums2:            
           
            zero_pos = nums2.index(0)
            if zero_pos == L-1:
                Lzeros = Lzeros
           
            else:
                next_val = nums2[zero_pos+1]
                while next_val == 0:
                    Lzeros = Lzeros + 1
                    if (zero_pos+Lzeros) > L-1:
                        break
                    else:
                        next_val = nums2[zero_pos + Lzeros]
           
            temp_arr = list(range(0, (zero_pos + Lzeros - 1 + zeros_removed)))
            temp_arr.reverse()
            diff_arr = [a-b for a,b in zip(nums[:(zero_pos + Lzeros - 1 
                                                  + zeros_removed)],temp_arr)]
           
            max_jump = max(diff_arr)
            if max_jump > 1 or \
                    (max_jump == 1 and (zero_pos + Lzeros == len(nums2))):
                del nums2[zero_pos:(zero_pos+Lzeros)]
                zeros_removed = zeros_removed + Lzeros
            else:
                return False
        return True

    def canJump2(self, nums):
        
        L = len(nums)
        num_zeros = nums.count(0)
        if L == 1:
            return True
       
        if nums[0] == 0:
            return False

        # If there are no zeros, then we can reach the end
        if num_zeros == 0:
            return True
       
        # Else, find index locations of right-most zero of all blocks of zeros
        zero_locs = []
        for i in range(0,L):
            if nums[i] == 0:
                if i == L-1:
                    zero_locs.append(i)
                elif nums[i+1] != 0:
                    zero_locs.append(i)
       
        sub_arr = list(range(0,L))
        sub_arr.reverse()
        for i in range(0,len(zero_locs)):
            sub_arr = list(range(0,zero_locs[i]))
            sub_arr.reverse()
           
            diff_arr = [a-b for a,b in zip(nums[:zero_locs[i]], sub_arr)]
            if zero_locs[i] == (L-1):
                if max(diff_arr[:zero_locs[i]]) <= 0:
                    return False
           
            elif max(diff_arr[:zero_locs[i]]) <= 1:
                return False
        return True
    
    def canJump3(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
       
        L = len(nums)
        num_zeros = nums.count(0)
       
        # If there are no zeros, then we can reach the end
        if num_zeros == 0:
            return True
       
        # Else, find index locations of right-most zero of all blocks of zeros
        zero_locs = []
        for i in range(0,L):
            if nums[i] == 0:
                if i == L-1:
                    zero_locs.append(i)
                elif nums[i+1] != 0:
                    zero_locs.append(i)
       
        sub_arr = list(range(0,L))
        sub_arr.reverse()
        for i in range(0,len(zero_locs)):
            sub_arr = list(range(0,zero_locs[i]))
            sub_arr.reverse()
           
            diff_arr = [a-b for a,b in zip(nums[:zero_locs[i]], sub_arr)]
            if zero_locs[i] == (L-1):
                if max(diff_arr[:zero_locs[i]]) <= 0:
                    return False
           
            elif max(diff_arr[:zero_locs[i]]) <= 1:
                return False
        return True
   
def main():
   
    #nums = [2,5,0,0]
    #nums = [2,3,1,1,4]
    #nums = [3,2,1,0,4]
    nums = [2,0,0]
    #nums = [3,0,8,2,0,0,1]
    #nums = [3,8,0,2,0,0,1]
    
    solution = Solution()
    result = solution.canJump2(nums)
    print("Reach end: ", result)
   
if __name__ == "__main__":
    main()
'''

###############################################################################
# LEETCODE PROBLEM: 45. Jump Game II (MEDIUM)
###############################################################################    
'''
def jump_helper(nums_abbr):      
    L = len(nums_abbr)
    for i in range(0,L):
        if ((i + nums_abbr[i]) - (L-1)) >= 0:
            return i
    return 0

class Solution(object):  
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        
        # Starting from the end, what is the largest jump you can make to
        # get to this position
        #
        # Repeat this from the location of the biggest jump to get you to the
        # end. Use helper function
       
        L = len(nums)
        pos = L-1
        jumps = 0
        while pos != 0:
            #pos = jump_helper(nums[:pos])
           
            nums_abbr = list(nums[:(pos+1)])
            L_abbr = len(nums_abbr)
            for i in range(0,L_abbr):
                if ((i + nums_abbr[i]) - (L_abbr-1)) >= 0:
                    pos = i
                    break
            jumps += 1
       
        return jumps
    
    def jump2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        L = len(nums)
        pos = L-1
        jumps = 0
        while pos != 0:
            #pos = jump_helper(nums[:pos])
           
            nums_abbr = list(nums[:(pos+1)])
            L_abbr = len(nums_abbr)
            for i in range(0,L_abbr):
                if ((i + nums_abbr[i]) - (L_abbr-1)) >= 0:
                    pos = i
                    break
            jumps += 1
        return jumps

def main():
    
    #nums = [2,5,0,0]
    #nums = [2,3,1,1,4]
    #nums = [3,2,1,0,4]
    nums = [2,0,0]
    #nums = [3,0,8,2,0,0,1]
    #nums = [3,8,0,2,0,0,1]
   
    solution = Solution()
    result = solution.jump(nums)
    print("Reach end: ", result)
   
if __name__ == "__main__":
    main()
'''

###############################################################################
# LEETCODE PROBLEM: 274. H-Index [MEDIUM]
###############################################################################
'''
class Solution(object):
    def hIndex(self, citations):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        citations.sort()
        citations.reverse()
        L = len(citations)
       
        h = 1
        for c in citations:
            if c >= h:
                h = h + 1
           
            else:
                break
       
        
        return h-1

def main():
   
    #citations = [1,3,1]
    citations = [3,0,6,1,5]
   
    solution = Solution()
    result = solution.hIndex(citations)
    print("H-index: ", result)
   
if __name__ == "__main__":
    main()
'''

###############################################################################
# LEETCODE PROBLEM: 238. Product of Array Except Self [MEDIUM]
###############################################################################
'''
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        L = len(nums)
        answer = [1 for _ in range(0,L)]
       
        prod_LR = 1
        prod_RL = 1
        # Set [i] element to product of all terms to left
        for i in range(0,L):
            answer[i] = prod_LR
            prod_LR = prod_LR * nums[i]

        # Set [i] element to product of all terms to right (times prev. result)
        for i in range(L-1,-1,-1):
            answer[i] = answer[i] * prod_RL
            prod_RL = prod_RL * nums[i]
       
        return answer
   
def main():
    
    nums = [3,0,6,1,5]
    
    solution = Solution()
    result = solution.productExceptSelf(nums)
    print(result)
   
if __name__ == "__main__":
    main()
'''

###############################################################################
# LEETCODE PROBLEM: 134. Gas Station [MEDIUM]
###############################################################################
'''
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
       
        L = len(gas)
       
        for i in range(0,L):

            j = i
            stops = 0
            tank = gas[i]
            while tank > 0:
               
                # Drive to next stop
                tank = tank - cost[j]
                if tank < 0:
                    break
               
                stops = stops + 1
               
                # If next stop wraps around, reset j, else increment
                if j == (L-1):
                    j = 0
                else:
                    j = j + 1
               
                # Fill up tank at next station
                tank = tank + gas[j]
               
                # If we are back to starting station, return True
                if stops == L:
                    return i
               
        return -1
               
def main():
    
    #gas = [1,2,3,4,5]
    #cost = [3,4,5,1,2]
    gas = [2,3,4]
    cost = [3,4,3]

    solution = Solution()
    print(solution.canCompleteCircuit(gas,cost))
   
if __name__ == "__main__":
    main()
'''

###############################################################################
# LEETCODE PROBLEM: 135. Candy [HARD]
###############################################################################
'''
class Solution(object):
    def candy(self, ratings):
       
        L = len(ratings)
        candies = [1 for i in range(0,L)]
       
        rSet = list(set(ratings))
        rSet.sort()
   
        for val in rSet:
            for i in range(0,L):
                if ratings[i] == val:
                   
                    # left edge case
                    if i == 0:
                        # if right-neighbor has lower ratings, increment by 1
                        if ratings[i+1] < val:
                            candies[i] = candies[i+1] + 1
                       
                    # right edge case
                    elif i == (L-1):
                        if ratings[i-1] < val:
                            candies[i] = candies[i-1] + 1
                   
                    # general case                            
                    else:
                        if ratings[i-1] < val or ratings[i+1] < val:
                            candies[i] = max(candies[i-1],candies[i+1]) + 1
           
        print(candies)
        total = 0
        for k in range(0,L):
            total = total + candies[k]
       
        return total
         
def main():
   
    ratings = [1,0,2]
    #ratings = [1,2,2]
    ratings = [1,2,87,87,87,2,1]
    ratings = [29,51,87,87,72,12]
    
    solution = Solution()
    print(solution.candy(ratings))
   
if __name__ == "__main__":
    main()
'''

###############################################################################
# LEETCODE PROBLEM: 151. Reverse Words in a String [MEDIUM]
###############################################################################
'''
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        words = s.split()
        reverse_s = words.pop()        
        for i in range(0,len(words)):
            reverse_s = reverse_s + " " + words.pop()
       
        return reverse_s
       
def main():
   
    #s = "the sky is blue"
    #s = "a good   example"
    s = "  hello world  "
    
    solution = Solution()
    print(solution.reverseWords(s))
   
if __name__ == "__main__":
    main()
'''

###############################################################################
# LEETCODE PROBLEM: 104. Maximum Depth of a Binary Tree [EASY]
###############################################################################
'''
import random

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0
        else:
            left_depth = self.maxDepth(root.left)
            right_depth = self.maxDepth(root.right)
            return 1 + max(left_depth, right_depth)

def createRandomTree(root, current_depth, max_depth):
    if current_depth < max_depth:
        left_val = random.randint(1,100)
        right_val = random.randint(1,100)
       
        root.left = TreeNode(left_val)
        root.right = TreeNode(right_val)
       
        createRandomTree(root.left, current_depth + 1, max_depth)
        createRandomTree(root.right, current_depth + 1, max_depth)

def main():
   
    # Create root node for tree
    root = TreeNode(1)
   
    # Create random rest of tree
    max_depth = 5
    createRandomTree(root, 1, max_depth)
   
    solution = Solution()
    print(solution.maxDepth(root))
   
if __name__ == "__main__":
    main()
'''

##############################################################################
# 45. Jump Game II [MEDIUM]
##############################################################################
'''
def jump_helper(nums_abbr):
       
        L = len(nums_abbr)
        for i in range(0,L):
            if ((i + nums_abbr[i]) - (L-1)) >= 0:
                return i
        return 0

class Solution(object):  
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # Starting from the end, what is the largest jump you can make to
        # get to this position
        #
        # Repeat this from the location of the biggest jump to get you to the
        # end. Use helper function
       
        L = len(nums)
        pos = L-1
        jumps = 0
        while pos != 0:
            #pos = jump_helper(nums[:pos])
           
            nums_abbr = list(nums[:(pos+1)])
            L_abbr = len(nums_abbr)
            for i in range(0,L_abbr):
                if ((i + nums_abbr[i]) - (L_abbr-1)) >= 0:
                    pos = i
                    break
            jumps += 1
       
        return jumps
       
def main():
    s = Solution()
    #nums = [2,0,0]
    nums = [3,2,8,0,0,0,0,1,2,4,0,0,1,0]
    #nums = [2,3,1,1,4]
    #nums = [1,2,3]
    #nums = [2,3,0,1,4]
    result = s.jump(nums)
    print(result)
   
if __name__ == "__main__":
    main()
'''

##############################################################################
# 36. Valid Sudoku [MEDIUM]
##############################################################################
'''
import numpy as np                

class Solution(object):  
    def isValidSudoku(self, board):
        """
        :type nums: List[int]
        :rtype: bool
        """
       
        # Check if each row contains unique entires
        for i in range(0,9):
            row_set = set()
            for j in range(0,9):
                curr = board[i][j]
                if curr != '.':
                    if curr in row_set:
                        return False
                    else:
                        row_set.add(curr)
                       
        # Check if each columns contains unique entries
        for i in range(0,9):
            col_set = set()
            for j in range(0,9):
                curr = board[j][i]
                if curr != '.':
                    if curr in col_set:
                        return False
                    else:
                        col_set.add(curr)
                       
        # Check if each 3x3 square contains unique entires
        x = [0,1,2]
        for h in range(0,3):
            y = [0,1,2]
            for i in range(0,3):
                square_set = set()
                for j in range(0,3):
                    for k in range(0,3):
                        curr = board[x[j]][y[k]]
                        if curr != '.':
                            if curr in square_set:
                                return False
                            else:
                                square_set.add(curr)
                y[0] += 3
                y[1] += 3
                y[2] += 3
            x[0] += 3
            x[1] += 3
            x[2] += 3
       
        return True

    def isValidSudoku_np(self, board):
        """
        :type nums: List[int]
        :rtype: bool
        """
       
        # Convert board to np.ndarray
        board_np = np.array(board)
       
        # Check if each row & column contains unique entires
        for i in range(0,9):
            uniques_row = len(set(board_np[i,:])) - 1
            ndots_row = np.sum(board_np[i,:] == '.')            
            if (uniques_row + ndots_row) < 9:
                return False
           
            uniques_col = len(set(board_np[:,i])) - 1
            ndots_col = np.sum(board_np[:,i] == '.')
            if (uniques_col + ndots_col) < 9:
                return False
           
        # Check if each 3x3 box contains unique entries
        x = 0
        for i in range(0,3):
            y = 0
            for j in range(0,3):
                curr_cube = board_np[x:x+3, y:y+3].tolist()
                curr_list = []
                curr_list.extend(curr_cube[0])
                curr_list.extend(curr_cube[1])
                curr_list.extend(curr_cube[2])
                curr_set = set(curr_list)
                uniques_box = len(curr_set) - 1
                ndots_box = np.sum(board_np[x:x+3, y:y+3] == '.')
                if (uniques_box + ndots_box) < 9:
                    return False
                y += 3
            x += 3
        return True
               
       
def main():

    board = [["5","3",".",".","7",".",".",".","."],
             ["6",".",".","1","9","5",".",".","."],
             [".","9","8",".",".",".",".","6","."],
             ["8",".",".",".","6",".",".",".","3"],
             ["4",".",".","8",".","3",".",".","1"],
             ["7",".",".",".","2",".",".",".","6"],
             [".","6",".",".",".",".","2","8","."],
             [".",".",".","4","1","9",".",".","5"],
             [".",".",".",".","8",".",".","7","9"]]
   
    board =  [["8","3",".",".","7",".",".",".","."],
              ["6",".",".","1","9","5",".",".","."],
              [".","9","8",".",".",".",".","6","."],
              ["8",".",".",".","6",".",".",".","3"],
              ["4",".",".","8",".","3",".",".","1"],
              ["7",".",".",".","2",".",".",".","6"],
              [".","6",".",".",".",".","2","8","."],
              [".",".",".","4","1","9",".",".","5"],
              [".",".",".",".","8",".",".","7","9"]]
    
    s = Solution()
    result = s.isValidSudoku_np(board)
    print(result)
   
if __name__ == "__main__":
    main()
'''
   
##############################################################################
# 228. Summary Ranges [EASY]
##############################################################################
'''
class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        ranges = []
        L = len(nums)

        if L == 0:
            return None
        if L == 1:
            ranges.append(str(nums[0]))
            return ranges
        
        left_num = str(nums[0])
        right_num = None
        for i in range(1,L):
            
            if left_num == None:
                left_num = str(nums[i])
            
            else:
                if nums[i] - nums[i-1] > 1:
                    if right_num:
                        ranges.append(left_num + "->" + right_num)
                    else:
                        ranges.append(left_num)
                    left_num = str(nums[i])
                    right_num = None
                
                else:
                    right_num = str(nums[i])
                    
            if i == (L-1):
                if right_num:
                    ranges.append(left_num + "->" + right_num)
                else:
                    ranges.append(left_num)
            
        return ranges
                    
def main():
    #nums = [0,1,2,4,5,7,11]
    #nums = [0,2,3,4,6,8,9]
    #nums = [-2147483648,-2147483647,2147483647]
    nums = [0,1,2,4,5,7]
    
    s = Solution()
    result = s.summaryRanges(nums)
    print(result)
   
if __name__ == "__main__":
    main()
'''

##############################################################################
# 121. Best Time to Buy and Sell Stock [EASY]
##############################################################################
'''
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        
        L = len(prices)
        b = 10000000000
        s = 0
        profit = 0
        
        for p in prices:
            b = min(b, p)
            s = max(s, p-b)
        
        return s
            

def main():
    #prices = [7,1,5,3,6,4]
    prices = [7,6,4,3,1]

    s = Solution()
    result = s.maxProfit(prices)
    print(result)
   
if __name__ == "__main__":
    main()
'''    
    
##############################################################################
# 122. Best Time to Buy and Sell Stock II [MEDIUM]
##############################################################################    
'''
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        
        L = len(prices)
        b = 10000000000
        s = 0
        profit = 0
        
        for p in prices:
            b = min(b, p)
            
            # if you can make a profit, sell and rebuy
            if p - b > 0:
                s = (p-b)
                profit += s
                
                # rebuy at current price
                b = p
            
        return profit
            

def main():
    prices = [7,1,5,3,6,4]
    #prices = [1,2,3,4,5]

    s = Solution()
    result = s.maxProfit(prices)
    print(result)
   
if __name__ == "__main__":
    main() 
'''
    
##############################################################################
# 123. Best Time to Buy and Sell Stock III [HARD]
##############################################################################    
'''
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) == 0:
            return 0
        
        b1 = 10**6
        b2 = 10**6
        s1 = 0
        s2 = 0

        for p in prices:
        
            b1 = min(b1, p)
            s1 = max(s1, p-b1)
            
            b2 = min(b2, p-s1)
            s2 = max(s2, p-b2)
        
        return s2

def main():
    prices = [3,3,5,0,0,3,1,4]
    #prices = [1,2,3,4,5]
    
    s = Solution()
    result = s.maxProfit(prices)
    print(result)
   
if __name__ == "__main__":
    main() 
'''    

##############################################################################
# 124. Best Time to Buy and Sell Stock IV [HARD]
##############################################################################    
'''
class Solution(object):
    def maxProfit(self, k, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) == 0 or k == 0:
            return 0
        
        b = [10**6]*k
        s = [0]*k

        for p in prices:
        
            b[0] = min(b[0], p)
            s[0] = max(s[0], p-b[0])
            
            for j in range(1,k): 
                b[j] = min(b[j], p-s[j-1])
                s[j] = max(s[j], p-b[j])
        
        return s[k-1]
        
def main():
    k = 2
    prices = [3,2,6,5,0,3]
    
    s = Solution()
    result = s.maxProfit(k,prices)
    print(result)
   
if __name__ == "__main__":
    main() 
'''    
##############################################################################
# 149. Max Points on a Line [HARD]
##############################################################################    
'''
import collections
import math
class Solution(object):
    def maxPoints(self, points):
        L = len(points)
        lines = [] #[m, b] of y = m*x + b
        nums = []
        
        if L <= 2:
            return L 
        
        for j in range(0,L):
            p1 = points[j]
            k = j+1
            while k < L:
                p2 = points[k]
                
                if p2[0] != p1[0]:
                    m = (p2[1]-p1[1]) / (p2[0]-p1[0])
                    b = p2[1] - m*p2[0]
                    line = [m,b]
                    if line not in lines:
                        lines.append(line)
                else: # vertical line, store lines as [x location, 10^6]
                    x_loc = p2[0]
                    line = [x_loc, 10**6]
                    if line not in lines:
                        lines.append(line)
                k += 1
        
        pts_per_line = [0]*len(lines)
        for j in range(0,L):
            k = 0
            for l in lines:
                p1 = points[j]
                if l[1] == 10**6: # if vertical line
                    if p1[0] == l[0]:
                        pts_per_line[k] += 1
                else: # not vertical line
                    diff = p1[1] - (l[0]*p1[0]+l[1])
                    if abs(diff) <= 0.000001:
                        pts_per_line[k] += 1

                k += 1
        return max(pts_per_line)

    def maxPoints2(self, points):
        L = len(points)
        if L <= 2:
            return L
        result = 2
        for i in range(0, L):
            cnt = collections.defaultdict(int)
            for j in range(0, L):
                if i != j:
                    angle = math.atan2(points[j][1]-points[i][1], points[j][0]-points[i][0])
                    cnt[angle] += 1
            result = max(result, max(cnt.values())+1)
        return result
def main():
    points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
    #points = [[4,5],[4,-1],[4,0]]
    #points = [[-6,-1],[3,1],[12,3]]
    
    
    s = Solution()
    result = s.maxPoints2(points)
    print(result)
    
if __name__ == "__main__":
     main()
'''
##############################################################################
# 189. Rotate Array [MEDIUM]
##############################################################################   

class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        L = len(nums)
        nums[0:L-k].reverse()
        nums[(L-k):L].reverse()
        nums.reverse()
        
    def rotate2(self, nums, k):
        
        pivot = len(nums)-(k)
        nums.extend(nums[:pivot])
        for i in range(0,pivot):
            del nums[0]
    
def main():
    nums = [1,2,3,4,5,6,7]
    k = 3
    s = Solution()
    s.rotate2(nums,k)
    print(nums)            
    
if __name__ == "__main__":
     main()
    
    
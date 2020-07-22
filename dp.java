/**
 * Index:62. Unique Paths(medium)
 * Time: 2020/5/5
 * Description: In matrix of m*n shape, a robot located at top-left corner trys reach bottom-right corner only in the way of 'right' or 'down';
 * Explanation: This is a typical 2D DP problem, we can store value in 2D DP array, but since we only need to use value at dp[i - 1][j] and dp[i][j - 1]
 * to update dp[i][j], we don't need to store the whole 2D table, but instead store value in an 1D array, and update data by using dp[j] = dp[j] + dp[j - 1],
 * (where here dp[j] corresponding to the dp[i - 1][j]) and dp[j - 1] corresponding to the dp[i][j - 1] in the 2D array)
*/

//Solution 1: 2D array //
class Solution {
    public int uniquePaths(int m, int n) {
        if(m==0 || n==0) return 0;
        if(m==1 || n==1) return 1;
        int[][] dp = new int[m][n];
        for(int i=0; i<m; i++){
            dp[i][0]=1;
        }
        for(int i=0; i<n; i++){
            dp[0][i]=1;
        }
        for(int i=1; i<m; i++){
            for(int j=1; j<n; j++){
                dp[i][j]=dp[i-1][j]+dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
}

// Solution 2: 1D array //
class Solution {
    public int uniquePaths(int m, int n) {
        if(m==0 || n==0) return 0;
        if(m==1 || n==1) return 1;
        int[] dp = new int[n];
        for(int i=0; i<n; i++){
            dp[i]=1;
        }
        for(int i=1; i<m; i++){
            for(int j=1; j<n; j++){
                dp[j]+=dp[j-1];
            }
        }
        return dp[n-1];
    }
}

/**
 * Index:63. Unique Paths II(medium)
 * Time: 2020/5/5
 * Description: In matrix of m*n shape, some obstacles are added to the grids;
 */

class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int width = obstacleGrid[0].length;
        int[] dp = new int[width];
        dp[0] = 1;
        for(int[] row: obstacleGrid){
            for(int j=0; j<width; j++){
                if(row[j]==1) dp[j]=0;
                else if(j>0) dp[j]+=dp[j-1];
            }
        }
        return dp[width-1];
    }
}

/**
 * 72. Edit Distance(hard)
 * Let following be the function definition :-
 * f(i, j) := minimum cost (or steps) required to convert first i characters of word1 to first j characters of word2
 * Case 1: word1[i] == word2[j], i.e. the ith the jth character matches.
 *     f(i, j) = f(i - 1, j - 1)
 * Case 2: word1[i] != word2[j], then we must either insert, delete or replace, whichever is cheaper
 *     f(i, j) = 1 + min { f(i, j - 1), f(i - 1, j), f(i - 1, j - 1) }
 *     f(i, j - 1) represents insert operation
 *     f(i - 1, j) represents delete operation
 *     f(i - 1, j - 1) represents replace operation
 * Here, we consider any operation from word1 to word2. It means, when we say insert operation, we insert a new character after word1 that matches the jth character of word2. So, now have to match i characters of word1 to j - 1 characters of word2. Same goes for other 2 operations as well.
 * Note that the problem is symmetric. The insert operation in one direction (i.e. from word1 to word2) is same as delete operation in other. So, we could choose any direction.
 * Above equations become the recursive definitions for DP.
 * Base Case:
 *     f(0, k) = f(k, 0) = k
 * Below is the direct bottom-up translation of this recurrent relation. It is only important to take care of 0-based index with actual code :-
 */
public class Solution {
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();

        int[][] cost = new int[m + 1][n + 1];
        for(int i = 0; i <= m; i++)
            cost[i][0] = i;
        for(int i = 1; i <= n; i++)
            cost[0][i] = i;

        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                if(word1.charAt(i) == word2.charAt(j))
                    cost[i + 1][j + 1] = cost[i][j];
                else {
                    int a = cost[i][j];
                    int b = cost[i][j + 1];
                    int c = cost[i + 1][j];
                    cost[i + 1][j + 1] = a < b ? (a < c ? a : c) : (b < c ? b : c);
                    cost[i + 1][j + 1]++;
                }
            }
        }
        // for(int i=0; i<cost.length; i++){
        //     for(int j=0; j<cost[0].length; j++){
        //         System.out.print(cost[i][j]);
        //     }
        //     System.out.println();
        // }
        return cost[m][n];
    }
}

/**
 * 91. Decode Ways(medium)
 */

class Solution {
    public int numDecodings(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) != '0' ? 1 : 0;
        for (int i = 2; i <= n; i++) {
            int first = Integer.valueOf(s.substring(i - 1, i));
            int second = Integer.valueOf(s.substring(i - 2, i));
            if (first >= 1 && first <= 9) {
                dp[i] += dp[i-1];
            }
            if (second >= 10 && second <= 26) {
                dp[i] += dp[i-2];
            }
        }
        return dp[n];
    }
}

/**
 * 70. Climbing Stairs(easy)
 */
class Solution {
    public int climbStairs(int n) {
        int[] dp = new int[n+1];
        if(n==1) return 1;
        if(n==2) return 2;
        dp[0]=0;
        dp[1]=1;
        dp[2]=2;
        for(int i=3; i<=n;i++){
            dp[i] = dp[i-2]+dp[i-1];
        }
        return dp[n];
    }
}

/**
 * 509. Fibonacci Number(easy)
 */
class Solution {
    public int fib(int N) {
        if(N==0) return 0;
        int[] f = new int[N+1];
        f[0]=0;
        f[1]=1;
        for(int i=2; i<=N; i++){
            f[i] = f[i-1]+f[i-2];
        }
        return f[N];
    }
}
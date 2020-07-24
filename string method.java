/**
 * String's method:
*/

/**
 *   (1)str.trim()
 *   删除前导空白和尾部空白。
 *   return String
 */

/**
 *   (2)str.split("\\s+")
 *   删除字符串内部的所有空格（不限个数），但不包括前导空白（包括尾部空白）
 *   return String[]
 */


/**
 * 49. Group Anagrams(medium)
 * (1) char[] a = str.toCharArray();  字符串转化为字符数组
 * (2) String s = String.valueOf(a);  字符数组转化为字符串
 * (3) 考察Map与List的应用，字符串与字符数组的转化；
 */
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        if(strs==null || strs.length==0) return new ArrayList<>();
        Map<String, List<String>> map = new HashMap<>();
        for(String s:strs){
            char[] a = s.toCharArray();
            Arrays.sort(a);
            String keyStr = String.valueOf(a);
            if(!map.containsKey(keyStr)) map.put(keyStr, new ArrayList<>());
            map.get(keyStr).add(s);
        }
        return new ArrayList<>(map.values());
    }
}

/**
 * 43. Multiply Strings(medium)
 * 这个题很棒，考察乘法的运算规则，很巧妙。
 * 我这是参考的discussion的代码，简洁清晰。
 */
class Solution {
    public String multiply(String num1, String num2) {
        int m=num1.length(), n=num2.length();
        int[] pos = new int[m+n];
        for(int i=m-1; i>=0; i--){
            for(int j=n-1; j>=0; j--){
                int mul = (num1.charAt(i)-'0')*(num2.charAt(j)-'0');
                int p1=i+j, p2=i+j+1;
                int sum = mul + pos[p2];

                pos[p1]+=sum/10;
                pos[p2]=sum%10;
            }
        }
        for (int i=0; i<m+n; i++){
            System.out.println(pos[i]);
        }
        StringBuilder sb = new StringBuilder();
        for(int p:pos) if(sb.length()!=0 || p!=0) sb.append(p);
        return sb.length()==0?"0":sb.toString();
    }
}
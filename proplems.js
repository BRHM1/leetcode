
function TreeNode(val, left, right) {
    this.val = (val === undefined ? 0 : val)
    this.left = (left === undefined ? null : left)
    this.right = (right === undefined ? null : right)
}

function ListNode(val) {
    this.val = val;
    this.next = null;
}

var deckRevealedIncreasing = function (deck) {
    const res = Array(deck.length)
    const queue = Array(deck.length)
    deck.sort((a, b) => a - b)
    for (let i = 0; i < queue.length; i++) queue[i] = i
    for (let card of deck) {
        let idx = queue.shift()
        res[idx] = card
        // skip the next index
        if (queue.length) queue.push(queue.shift())
    }
    return res
};
var removeKdigits = (num, k) => {
    let stack = []
    if (num.length === k) return "0"
    for (let n of num) {
        while (stack.at(-1) > n && k > 0) {
            stack.pop();
            k--
        }
        stack.push(n)
        while (stack.at(-1) === "0" && stack.length === 1) stack.pop()
    }
    return stack.slice(0, stack.length - k).join('') || "0"
}

var trap = function (height) {
    let water = 0, units = 0, diff = 0
    let stack = []
    let flag = false
    for (let i = 0; i < height.length; i++) {
        while (units) {
            stack.push(stack.at(-1))
            units--
        }
        units = 0
        while (height[i] > stack.at(-1)) {
            flag = true
            if (stack.at(-2) < stack.at(-1) || stack.length === 1) break
            diff += stack.pop()
            units++
        }
        if (flag) {
            water += Math.min(height[i], stack.at(-1)) * units - diff
        }
        stack.push(height[i])
        diff = 0
        flag = false
    }
    return water
};

const deepCopy = (obj) => {
    let res = {}
    for (let key in obj) {
        if (typeof obj[key] === "object") {
            res[key] = deepCopy(obj[key])
        } else {
            res[key] = obj[key]
        }
    }
    return res
}

var maximalRectangle = function (matrix) {
    // approach : combine each row with the row on top of it to create histogram then calculate the max area in that histogram
    // if there is a zero it makes the total col = 0
    let max = 0
    let histogram_arr = Array(matrix.length).fill().map(() => Array(matrix[0].length))
    for (let r = 0; r < matrix.length; r++) {
        for (let c = 0; c < matrix[0].length; c++) {
            if (r === 0) {
                histogram_arr[r][c] = +matrix[r][c]
            } else if (matrix[r][c] === "0") {
                histogram_arr[r][c] = 0
            } else {
                histogram_arr[r][c] = +histogram_arr[r - 1][c] + 1
            }
        }
    }
    for (let histogram of histogram_arr) {
        max = Math.max(max, largestRectangleArea(histogram))
    }
    return max
};

const largestRectangleArea2 = (histogram) => {
    let maxArea = 0
    let stack = [] // [starting index , height]
    for (let i = 0; i < histogram.length; i++) {
        let start = i
        while (stack.length && stack.at(-1)[1] > histogram[i]) { // it means i can't extend it further more 
            let [index, height] = stack.pop()
            maxArea = Math.max(maxArea, (i - index) * height)
            // the popped one is bigger than the current one , it means i can extend the current one to the left direction
            // so i made the start of the current one === to the index i just popped
            start = index
        }
        stack.push([start, histogram[i]])
    }
    // may be some element's left in the stack (those who made it to the end of the histogram)
    for (let [index, height] of stack) {
        maxArea = Math.max(maxArea, (histogram.length - index) * height)
    }
    return maxArea
}

var sumOfLeftLeaves = function (root) {
    let sum = 0
    const helper = (node, key) => {
        if (!node) return 0
        if (!node.left && !node.right && key) sum += node.val
        helper(node.left, true)
        helper(node.right, false)
    }
    helper(root, false)
    return sum
};


var sumNumbers = function (root, current = 0) {
    let sum = 0
    if (!root) return 0
    if (!root.left && !root.right) sum += +(current * 10 + root.val)
    sum += sumNumbers(root.left, current * 10 + root.val) + sumNumbers(root.right, current * 10 + root.val)
    return sum
};

var addOneRow = function (root, val, depth) {
    if (!root) return;
    if (depth === 2) {
        let oldLeft = root.left
        let oldRight = root.right
        root.left = new TreeNode(val, oldLeft, null)
        root.right = new TreeNode(val, null, oldRight)
        return root
    }
    if (depth === 1) {
        const node = new TreeNode(val, root)
        return node
    }
    root.left = addOneRow(root.left, val, depth - 1)
    root.right = addOneRow(root.right, val, depth - 1)
    return root;
};


var smallestFromLeaf = function (root) {
    let smallest = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
    const helper = (node, current) => {
        if (!node) return
        if (!node.left && !node.right) {
            smallest = String.fromCharCode(97 + +node.val) + current > smallest ? smallest : String.fromCharCode(97 + +node.val) + current
            return
        }
        helper(node.left, String.fromCharCode(97 + +node.val) + current)
        helper(node.right, String.fromCharCode(97 + +node.val) + current)
    }
    helper(root, "")
    return smallest
};

var islandPerimeter = function (grid) {
    let res = 0
    for (let row = 0; row < grid.length; row++) {
        for (let col = 0; col < grid[0].length; col++) {
            if (grid[row][col] === 1) {
                // look at the four directions if the neighbor is land increment the res by 1
                res += 4
                if (row !== 0) res -= grid[row - 1][col] === 1 ? 1 : 0
                if (row !== grid.length - 1) res -= grid[row + 1][col] === 1 ? 1 : 0
                if (col !== 0) res -= grid[row][col - 1] === 1 ? 1 : 0
                if (col !== grid[0].length - 1) res -= grid[row][col + 1] === 1 ? 1 : 0
            }
        }
    }
    return res
};

var numIslands = function (grid) {
    let res = 0
    const dfs = (r, c) => {
        if (r < 0 || r >= grid.length || c < 0 || c >= grid[0].length) return 0
        if (grid[r][c] === "0") return 0

        grid[r][c] = "0"

        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
        return 1
    }
    for (let r = 0; r < grid.length; r++) {
        for (let c = 0; c < grid[0].length; c++) {
            if (grid[r][c] === "1") res += dfs(r, c)
        }
    }
    return res
};


var findFarmland = function (land) {
    const res = []
    const ROWS = land.length, COLS = land[0].length

    const dfs = (r, c, edges) => {
        if (r < 0 || r >= ROWS || c < 0 || c >= COLS) return true;
        if (land[r][c] === 0) return true
        if (land[r][c] === null) return false

        land[r][c] = null
        let isRightEdge = dfs(r + 1, c, edges)
        let isDownEdge = dfs(r, c + 1, edges)

        if (isDownEdge && isRightEdge) {
            edges.row = r
            edges.col = c
        }
    }
    for (let i = 0; i < land.length; i++) {
        for (let j = 0; j < land[0].length; j++) {
            if (land[i][j] !== 1) continue
            const edges = { row: null, col: null }
            dfs(i, j, edges)
            let subResult = [i, j, edges.row, edges.col]
            res.push(subResult)
        }
    }
    return res
}

// var validPath = function (n, edges, source, destination) {
//     const adjacencyList = {}
//     if(source === destination) return true
//     if(!edges.length) return false
//     for (let [src, dis] of edges) {
//         adjacencyList[src] = adjacencyList[src] ? [...adjacencyList[src], dis] : [dis]
//         adjacencyList[dis] = adjacencyList[dis] ? [...adjacencyList[dis], src] : [src]
//     }
//     // apply bfs to find if a path exists
//     const queue = [source]
//     const visited = new Set()
//     while (queue.length) {
//         let current = queue.shift()
//         for (let neighbor of adjacencyList[current]) {
//             if (neighbor === destination) return true
//             if (!visited.has(neighbor)) {
//                 queue.push(neighbor)
//                 visited.add(neighbor)
//             }
//         }
//     }
//     return false
// };

class UnionFind {
    parent = new Map()
    edges = []
    constructor(n, edges) {
        this.parent = new Map(Array.from(Array(n).keys()).map(i => [i, i]))
        this.edges = edges
    }
    find(node) { // returns the representive of the group for a specific node
        let root = node
        while (root !== this.parent.get(root)) {
            root = this.parent.get(root)
        }
        return root
    }
    union(group1, group2) {
        let group1Representive = this.find(group1)
        let group2Representive = this.find(group2)
        this.parent.set(group1Representive, group2Representive)
    }
    populate() {
        for (let [src, des] of this.edges) {
            this.union(src, des)
        }
        return this.parent
    }
}

const validPath = (n, edges, source, destination) => {
    const nodes = new Map(Array.from(Array(n).keys()).map(i => [i, i])) // make each node is the parent(representive) of itself
    const find = (node) => { // find's the representive of the input node 
        let root = node
        while (root !== nodes.get(root)) {
            root = nodes.get(root)
        }
        return root
    }
    const union = (firstNode, secondNode) => { // group's two trees together under one root 
        let firstNodeRoot = find(firstNode)
        let secondNodeRoot = find(secondNode)
        nodes.set(firstNodeRoot, secondNodeRoot)
    }
    const isConnected = (first, second) => {
        return find(first) === find(second)
    }
    for (let [src, des] of edges) {
        union(src, des)
    }
    return isConnected(source, destination)
}

var openLock = function (deadends, target) {
    const queue = [["0000", 0]]
    const dead = new Set(deadends)
    const visited = new Set(deadends)

    while (queue.length) {
        let [num, minMoves] = queue.shift()
        if (num === target) return minMoves
        if (dead.has(num)) continue
        for (let neighbor of generateNeighbors(num)) {
            if (!visited.has(neighbor)) {
                visited.add(neighbor)
                queue.push([neighbor, minMoves + 1])
            }
        }
    }
    function generateNeighbors(num) {
        const res = []
        for (let i = 0; i < num.length; i++) {
            res.push(num.slice(0, i) + ((+num[i] + 1) % 10) + num.slice(i + 1))
            res.push(num.slice(0, i) + ((+num[i] + 9) % 10) + num.slice(i + 1))
        }
        return res
    }
    return -1
};


var findMinHeightTrees = function (n, edges) {
    // intiution : remove the leaf nodes 
    const adj = {}
    const edges_count = {}
    const leaves = [] // queue
    if (edges.length === 0) return [0]
    for (let [src, des] of edges) {
        adj[src] = adj[src] ? [...adj[src], des] : [des]
        adj[des] = adj[des] ? [...adj[des], src] : [src]
    }
    for (let node in adj) {
        let neighbors = adj[node]
        if (neighbors.length === 1) leaves.push(node)
        edges_count[node] = neighbors.length
    }
    while (leaves.length) {
        let len = leaves.length
        if (n <= 2) {
            return leaves
        }
        for (let i = 0; i < len; i++) {
            let node = leaves.shift()
            n -= 1
            for (let neighbor of adj[node]) {
                console.log(neighbor)
                edges_count[neighbor] -= 1
                if (edges_count[neighbor] === 1) leaves.push(neighbor)
            }
        }
    }
    return leaves
};

var findMinHeightTrees2 = (n, edges) => {
    let furthest = 0
    let height = 0
    const adj = new Map()
    let path = []
    if (!edges.length) return [0]
    for (let [src, des] of edges) {
        adj[src] = adj[src] ? [...adj[src], des] : [des]
        adj[des] = adj[des] ? [...adj[des], src] : [src]
    }
    let visited = new Set()
    const dfs = (root, depth, goal) => {
        if (root === goal) return [...path, root]
        if (root === undefined || visited.has(root)) return []
        if (depth > height) {
            height = depth
            furthest = root
        }
        visited.add(root)
        path.push(root)
        for (let neighbor of adj[root]) {
            const result = dfs(neighbor, depth + 1, goal)
            if (result.length > 0) return result
        }
        path.pop(root)
        return []
    }


    dfs(Math.floor(Math.random() * 10) % n, 0, Infinity)
    let src = furthest
    visited = new Set()
    height = 0

    dfs(furthest, 0, Infinity)
    let des = furthest
    visited = new Set()
    height = 0
    console.log(src, des)
    let p = dfs(src, 0, des)
    while (p.length > 2) {
        p.shift()
        p.pop()
    }
    return p
}

var combinationSum = function (candidates, target) {
    let res = []
    const dfs = (curSum, curPath, candidates) => {
        if (curSum > target) return
        if (curSum === target) return 1 && res.push([...curPath])
        for (let i = 0; i < candidates.length; i++) {
            curPath.push(candidates[i])
            dfs(curSum + candidates[i], curPath, candidates.slice(i))
            curPath.pop()
        }
    }
    dfs(0, [], candidates)
    return res
};

var combinationSum2 = function (candidates, target) {
    const res = []
    candidates.sort((a, b) => a - b)
    const dfs = (cur, index, target) => {
        if (target < 0) return
        if (0 === target) return 1 && res.push([...cur])
        let prev = -1
        for (let i = index; i < candidates.length; i++) {
            if (candidates[i] === prev) continue
            cur.push(candidates[i])
            dfs(cur, i + 1, target - candidates[i])
            cur.pop()
            prev = candidates[i]
        }
    }
    dfs([], 0, target)
    return res
};


var tribonacci = function (n) {
    let f = 0, s = 1, t = 1
    if (n === 0) return 0
    for (let i = 2; i < n; i++) {
        let temp = f + s + t
        f = s
        s = t
        t = temp
    }
    return t
};

var subsets = function (nums) {
    const res = [[]]
    const helper = (subset, nums) => {
        if (!nums.length) return []
        for (let i = 0; i < nums.length; i++) {
            subset.push(nums[i])
            res.push([...subset])
            helper(subset, nums.slice(i + 1))
            subset.pop()
        }
    }
    helper([], nums)
    return res
};

var subsetsWithDup = function (nums) {
    let res = [[]]
    nums.sort((a, b) => a - b)

    const backtrack = (index, subset) => {
        if (index >= nums.length) return
        let prev = null
        for (let i = index; i < nums.length; i++) {
            if (nums[i] == prev) continue
            subset.push(nums[i])
            res.push([...subset])
            backtrack(i + 1, subset)
            subset.pop(nums[i])
            prev = nums[i]
        }
    }
    backtrack(0, [])
    return res
};


// invented new algorithm
// 1- intialize dp array of length s and fill it with zeros
// 2- iterate over the string from the right to left 
// 3- first element from the right has a maximum length of 1 
//   (because no charachters comes after it so it will be subsequence of length 1) 
// 4- so we intialize the dp.at(-1) with 1 , max[s.at(-1)] = 1 , onRight = {s.at(-1)}
// 5- we keep a max hashtable with length 26 (each char assigned to 0 intially) to keep track of what is the longest subsequence from that char till the end of the string
// 6- keep a set with all elements that found on right to make sure we calculate only the ones we encountered and helping in not clashing with the repeated characters
// 8- begin from s.length -2 then look to the right => is there a char which asciiDifference(currChar - thatChar) <= k && onRight.has(thatChar) ?
//    if yes then do the following => dp[i] = Math.max(dp[i] , max[char] + 1) , max[s[i]] = (dp[i] || 1) , onRight.add(s[i])
// 9- after doing this operation for all the elements on right of s[i]:
//           1) modify max[s[i]] to be equall to (dp[i] || 1) 
//           2) onRight.add(s[i])
//10- keep doing this untill you fill the dp array then pick the maximum

// time = O(26n) => O(n)
// space = O(s.length + 26 + 26) => O(n)

// EXAMPLE: longestIdealString("abczzzca" , 2) => dp = [5,4,3,3,2,1,2,1]


var longestIdealString = function (s, k) {
    const dp = Array(s.length).fill(0)
    dp[dp.length - 1] = 1
    let max = new Map(Array.from(Array(26), (_, i) => [String.fromCharCode(i + 97), 0]))
    const onRight = new Set(s.at(-1))
    max.set(s[s.length - 1], 1)
    for (let i = dp.length - 2; i >= 0; i--) {
        const asciiForSChar = s[i].charCodeAt(0)
        for (let [key, value] of max) {
            const asciiForMaxChar = key.charCodeAt(0)
            let diff = Math.abs(asciiForMaxChar - asciiForSChar)
            if (diff <= k && value >= 1 && onRight.has(key)) {
                dp[i] = Math.max(dp[i], value + 1)
            }
        }
        max.set(s[i], dp[i] || 1)
        onRight.add(s[i])
    }
    return dp
}

var minFallingPathSum = function (grid) {
    // intialize empty grid and fill each row with the minimum it can be (BASE + Math.min(row - 1 && !same column))
    let ROWS = grid.length, COLS = grid[0].length
    let dp = Array(COLS).fill(0)
    const getLowestValues = (arr) => {
        let lowest = [Infinity, 0], secondLowest = [Infinity, 0]
        for (let i = 0; i < COLS; i++) {
            if (arr[i] < lowest[0]) {
                secondLowest = lowest
                lowest = [arr[i], i]
            }
            if (arr[i] > lowest[0] && arr[i] < secondLowest[0]) secondLowest = [arr[i], i]
            if (arr[i] === lowest[0] && lowest[1] !== i) secondLowest = [arr[i], i]
        }
        return [lowest, secondLowest]
    }
    let [lowest, secondLowest] = getLowestValues(grid[0])
    for (let r = 1; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (c !== lowest[1]) {
                dp[c] = grid[r][c] + lowest[0]
            } else {
                dp[c] = grid[r][c] + secondLowest[0]
            }
        }
        [lowest, secondLowest] = getLowestValues(dp)
        dp = Array(COLS).fill(0)
    }
    return lowest[0]
};


var findRotateSteps = function (ring, key) {
    let left = i => i === 0 ? ring.length - 1 : i - 1;
    let right = i => i === ring.length - 1 ? 0 : i + 1;
    let dp = ring.split("").map(() => 0);

    for (let i = key.length - 1; i >= 0; i--) {
        let dp1 = ring.split("").map((x, j) => (x === key[i]) ? dp[j] : Infinity);
        for (let j = 0; j < ring.length * 2; j++) {
            let x = j % ring.length;
            dp1[x] = Math.min(dp1[x], dp1[left(x)] + 1);
            let y = ((ring.length * 2) - 1 - j) % ring.length;
            dp1[y] = Math.min(dp1[y], dp1[right(y)] + 1);
        }
        dp = dp1;
    }
    return dp[0] + key.length;
};


var sumOfDistancesInTree = function (n, edges) {
    const graph = new Array(n).fill(null).map(() => []);
    const count = new Array(n).fill(0);
    const res = new Array(n).fill(0);
    for (const [u, v] of edges) {
        graph[u].push(v);
        graph[v].push(u);
    }
    console.log(graph, count)

    const dfs1 = (cur, parent) => {
        count[cur] = 1;
        for (const child of graph[cur]) {
            if (child !== parent) {
                dfs1(child, cur);
                count[cur] += count[child];
                res[cur] += res[child] + count[child];
            }
        }
    };

    const dfs2 = (cur, parent) => {
        for (const child of graph[cur]) {
            if (child !== parent) {
                res[child] = res[cur] + n - 2 * count[child];
                dfs2(child, cur);
            }
        }
    };

    dfs1(0, -1);
    dfs2(0, -1);

    return res;
};

function wonderfulSubstrings(word) {
    let count = 0;
    const n = word.length;
    const freq = new Array(1024).fill(0); // Array to store frequencies of characters
    freq[0] = 1; // Initialize with an empty substring

    let bitmask = 0; // Bitmask to represent frequency of characters

    // Iterate over all characters
    for (let i = 0; i < n; i++) {
        const charIndex = word.charCodeAt(i) - 'a'.charCodeAt();
        bitmask ^= 1 << charIndex; // Toggle the bit for the current character

        // Increase count for wonderful substrings
        count += freq[bitmask];

        // Update frequencies array
        for (let j = 0; j < 10; j++) {
            const newBitmask = bitmask ^ (1 << j);
            count += freq[newBitmask];
        }

        freq[bitmask]++;
    }

    return count;
}

const determineSpecificBit = (number, n) => {
    // this function checks if a specific bit is equall to 0 or 1 by using bitmasking
    let res = number.toString(2)
    let bitmask = (1 << (n - 1)).toString(2).padStart(res.length, 0)
    return (bitmask & res) === 0 ? 0 : 1
}

var reversePrefix = function (word, ch) {
    let idx = word.indexOf(ch)
    let post = word.slice(idx + 1)
    let reversedPre = word.slice(0, idx + 1)
    return idx ? reversedPre.split('').reverse().join('') + post : word
};


var findMaxK = function (nums) {
    let isFound = new Set(nums)
    let max = 0
    for (let num of nums) {
        max = isFound.has(num * -1) ? Math.max(Math.abs(num), max) : max
    }
    return max || -1
};

var compareVersion = function (version1, version2) {
    let v1 = version1.split(".")
    let v2 = version2.split(".")
    for (let i = 0; i < Math.max(v1.length, v2.length); i++) {
        let num1 = i < v1.length ? parseInt(v1[i]) : 0
        let num2 = i < v2.length ? parseInt(v2[i]) : 0
        if (num1 > num2) return 1
        if (num1 < num2) return -1
    }
    return 0
};

var deleteNode = function (node) {
    node.val = node.next.val
    node.next = node.next.next
};

var doubleIt = function (head) {
    // approach : reverse the linked list and start for node.val * 2 + remainder from the previous operation
    const reverse = head => {
        let prev = null, cur = head
        while (cur !== null) {
            let temp = cur.next
            cur.next = prev
            prev = cur
            cur = temp
        }
        return prev
    }
    let dummy = reverse(head)
    let res = dummy
    let remainder = 0
    while (dummy !== null) {
        let num = ((2 * dummy.val) % 10) + remainder
        remainder = Math.floor(2 * dummy.val / 10)
        dummy.val = num
        dummy = dummy.next
    }
    return reverse(res)
};

var findRelativeRanks = function (score) {
    let tuple = ["Gold Medal", "Silver Medal", "Bronze Medal"]
    let res = Array(score.length)
    let pairs = score.map((val, i) => [val, i])
    pairs.sort((a, b) => b[0] - a[0])
    for (let i = 0; i < score.length; i++) {
        res[pairs[i][1]] = i > 2 ? `${i + 1}` : tuple[i]
    }
    return res
};

var maximumHappinessSum = function (happiness, k) {
    happiness.sort((a, b) => b - a)
    let sum = happiness[0]
    let decrementCount = 1
    for (let i = 1; i < k; i++) {
        if (happiness[i] - decrementCount > 0) sum += happiness[i] - decrementCount
        decrementCount++
    }
    return sum
};

var kthSmallestPrimeFraction = function (arr, k) {
    let l = 0, n = k, r = arr.length - 1, min = arr[l] / arr[r]
    let res = [[arr[l], arr[r]]]
    while (k > 0) {
        // you can decrease the right ptr or increase the left one 
        if (r - l > 1 && k > 0) {
            let leftInc = arr[l + 1] / arr[r]
            let rightDec = arr[l] / arr[r - 1]
            if (leftInc < rightDec) {
                res.push([arr[l + 1], arr[r]], [arr[l], arr[r - 1]])
                l++
            } else {
                res.push([arr[l], arr[r - 1]], [arr[l + 1], arr[r]])
                r--
            }
        }
        k--
    }
    return res[n - 1] || res.at(-1)
};

var largestLocal = function (grid) {
    let res = Array(grid.length - 2).fill().map(() => Array(grid[0].length - 2))
    const getMaxOutOfRow = row => {
        let modifiedRow = []
        let window = []
        for (let i = 0; i < row.length; i++) {
            window.push(row[i])
            if (window.length == 3) {
                modifiedRow.push(Math.max(...window))
                window.shift()
            }
        }
        return modifiedRow
    }
    let newgrid = []
    for (let i = 0; i < grid.length; i++) {
        newgrid.push(getMaxOutOfRow(grid[i]))
    }
    for (let i = 0; i < newgrid[0].length; i++) {
        let colwindow = []
        for (let j = 0; j < newgrid.length; j++) {
            colwindow.push(newgrid[j][i])
            if (colwindow.length === 3) {
                res[j % grid[0].length - 2][i] = Math.max(...colwindow)
                colwindow.shift()
            }
        }
    }
    return res
};

var matrixScore = function (grid) {
    // trying to maximize the first col to be all one's
    let sum = 0
    let colsToFlip = Array(grid[0].length).fill(0)
    for (let row = 0; row < grid.length; row++) {
        let flip = false
        for (let col = 0; col < grid[0].length; col++) {
            // check if the first cell in the row == 0 if it is flip the whole row
            // for each col look at the number of one's if it's less than 0's in that col flip the col
            if (grid[row][col] == 0 && col == 0) flip = true
            if (flip) grid[row][col] = 1 - grid[row][col]
            if (!grid[row][col] && colsToFlip[col] != "*") {
                colsToFlip[col] += 1
                if (colsToFlip[col] > grid.length / 2) colsToFlip[col] = "*"
            }
        }
        flip = false
    }
    let start = 1
    let binaryMap = Array.from(Array(grid[0].length), (elm) => {
        elm = start * 2
        start *= 2
        return elm
    })
    binaryMap.unshift(1)
    binaryMap.pop()
    binaryMap.reverse()
    for (let i = 0; i < grid.length; i++) {
        for (let j = 0; j < grid[0].length; j++) {
            if (colsToFlip[j] === '*') {
                sum += (1 - grid[i][j]) * binaryMap[j]
            } else {
                sum += grid[i][j] * binaryMap[j]
            }
        }
    }
    return sum
};

// var getMaximumGold = function(grid) {
//     const ROWS = grid.length , COLS = grid[0].length
//     let max = -Infinity
//     let curSum = 0
//     const getDirections = (r , c) => {
//         const res = []
//         if(c + 1 < COLS) res.push([r , c + 1])
//         if(c - 1 >= 0) res.push([r , c - 1])
//         if(r + 1 < ROWS) res.push([r + 1, c])
//         if(r - 1 >= 0) res.push([r - 1, c])
//         return res
//     }
//     console.log(getDirections(2,2))
//     // run dfs backtrack on the grid
//     const dfs = (r , c , visited) => {
//         if(grid[r][c] === 0 || visited.has(`${r,c}`)) return 0
//         let tmp = grid[r][c]
//         // grid[r][c] = 0 // add to visited
//         visited.add(`${r,c}`)
//         curSum += tmp
//         max = Math.max( max , curSum )
//         for(let [row , col] of getDirections(r , c)){
//             dfs(row , col, visited)
//         }
//         curSum -= tmp
//         // grid[r][c] = tmp // undo from visited
//         visited.delete(`${r,c}`)
//     }
//     for(let i = 0; i < ROWS; i++){
//         for(let j = 0; j < COLS; j++){
//             if(grid[i][j] === 0) continue
//             curSum = 0
//             dfs(i , j, new Set())
//         }
//     }
//     return max
// };

var getMaximumGold = (grid) => {
    const ROWS = grid.length, COLS = grid[0].length

    const dfs = (r, c, visited) => {
        if (Math.min(r, c) < 0 || r == ROWS ||
            c == COLS || grid[r][c] == 0 ||
            visited.has(`${r},${c}`)) return 0
        visited.add(`${r},${c}`)
        console.log(visited)
        res = grid[r][c]
        const neighbors = [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]]
        for (let [row, col] of neighbors) {
            res = Math.max(res, grid[r][c] + dfs(row, col, visited))
        }
        visited.delete(`${r},${c}`)
        return res
    }
    let res = 0
    for (let i = 0; i < ROWS; i++) {
        for (let j = 0; j < COLS; j++) {
            if (grid[i][j] === 0) continue
            res = Math.max(res, dfs(i, j, new Set()))
        }
    }
    return res
}

var evaluateTree = function (root) {
    if (!root.left) return !!root.val
    let res = false
    if (root.val === 2) {
        res = evaluateTree(root.left) || evaluateTree(root.right)
    } else if (root.val === 3) {
        res = evaluateTree(root.left) && evaluateTree(root.right)
    }
    return !!res
};

var removeLeafNodes = function (root, target) {
    const post_order = (root, target) => {
        if (!root) return null
        root.left = post_order(root.left, target)
        root.right = post_order(root.right, target)
        if (root.left === root.right && root.val === target) return null
        return root
    }
    return post_order(root, target)
};

var distributeCoins = function (root) {
    let sum = 0
    // note that there are n nodes also n coins
    const helper = (root) => {
        // if it's a leaf node it should send to the root -val or +val
        if (!root) return 0
        if (!root.left && !root.right) return root.val - 1
        let left = helper(root.left)
        let right = helper(root.right)
        sum += Math.abs(left) + Math.abs(right)
        return root.val + left + right - 1
    }
    helper(root)
    return sum
};

// 3068. Find the Maximum Sum of Node Values
var maximumValueSum = function (nums, k, edges) {
    let delta = Array.from(nums, num => ((num ^ k) - num))
    delta.sort((a, b) => b - a)
    let sum = nums.reduce((acc, cur) => acc + cur, 0)
    for (let i = 0; i < nums.length; i += 2) {
        if (i + 1 >= nums.length) break
        if (delta[i] + delta[i + 1] <= 0) break
        sum += delta[i] + delta[i + 1]
    }
    return sum
};

var subsetXORSum = function (nums) {
    let sum = 0
    const backtrack = (cur_index, xor) => {
        if (cur_index >= nums.length) {
            sum += xor
            return
        }
        xor ^= nums[cur_index]
        for (let i = cur_index; i < nums.length; i++) backtrack(i + 1, xor)
        xor ^= nums[cur_index]
    }
    for (let i = 0; i < nums.length; i++) backtrack(i, 0)
    return sum
};

var subsetXORSum = function (nums) {
    return nums.reduce((acc, cur) => acc | cur) << nums.length - 1
};

var subsetsII = function (nums) {
    const paths = [[]]
    const backtrack = (cur_index, path) => {
        if (cur_index >= nums.length) return 1 && paths.push([...path])
        path.push(nums[cur_index])
        for (let i = cur_index; i < nums.length; i++) {
            backtrack(i + 1, path)
        }
        path.pop()
    }
    for (let i = 0; i < nums.length; i++) backtrack(i, [])
    return paths
};



const partition = s => {
    const res = []
    const part = []
    const isPlindrome = (str, l, r) => {
        while (l <= r) {
            if (str[l] !== str[r]) return false
            l++
            r--
        }
        return true
    }
    const backtrack = (idx) => {
        if (idx >= s.length) return 1 && res.push([...part])
        for (let i = idx; i < s.length; i++) {
            if (isPlindrome(s, idx, i)) {
                part.push(s.slice(idx, i + 1))
                backtrack(i + 1)
                part.pop()
            }
        }
    }
    backtrack(0)
    return res
}


var maxScoreWords = function (words, letters, score) {
    // backtrack each subset i can take and know the global count at one moment 
    // if i exceed the global count it's no more valid path
    let res = 0
    let count = new Map()
    for (let letter of letters) {
        count.set(letter, (count.get(letter) || 0) + 1)
    }
    let word_values = new Map()
    for (let word of words) {
        let res = 0
        for (let letter of word) {
            res += score[letter.charCodeAt(0) - 97]
        }
        word_values.set(word, res)
    }
    console.log(word_values)
    const paths = []
    const backtrack = (idx, current_res, path) => {
        // determine if this is a valid path if not return
        if (idx >= words.length) {
            res = Math.max(current_res, res)
            return 1 && paths.push([...path])
        }
        let temp = word_values.get(words[idx])
        let temp_count = new Map()
        const isValidPath = () => {
            for (let letter of words[idx]) {
                if (count.has(letter) === false) return false
                count.set(letter, count.get(letter) - 1);
                temp_count.set(letter, (temp_count.get(letter) || 0) + 1);
                // the path is no more valid 
                if (count.get(letter) < 0) return false
            }
            return true
        }
        // add to current result
        let valid = isValidPath()
        if (valid) current_res += temp
        path.push(words[idx])
        backtrack(idx + 1, current_res, path)
        path.pop()
        // remove the added value to current result
        if (valid) current_res -= temp
        // add the decremented count to the path

        for (let [letter, value] of temp_count) {
            count.set(letter, count.get(letter) + value)
        }

        backtrack(idx + 1, current_res, path)
    }
    backtrack(0, 0, [])
    return res
};

const sayed = (machines, inp) => {
    let res = []
    // console.log(inp.length)
    for (let i = 0; i < machines / 2; i += 2) {
        let min = Infinity
        let needed = inp[i]
        let bank = inp[i + 1]
        for (let j = 0; j < 3; j++) {
            min = Math.min(min, Math.floor(bank[j] / needed[j]))
        }
        res.push(min)
    }
    return res
}

const hard_sayed = (T, inp) => {
    let [rcd, bank, price, total] = inp
    let new_capacity = []
    let res = sayed(1, [rcd, bank])
    for (let i = 0; i < 3; i++) {
        new_capacity[i] = bank[i] - (res[0] * rcd[i])
    }
    while (total > 0) {
        let cost = 0
        for (let i = 0; i < 3; i++) {
            if (new_capacity[i] >= rcd[i]) {
                new_capacity[i] -= rcd[i]
                continue
            }
            let units = rcd[i] - new_capacity[i]
            let unit_price = price[i]
            cost += unit_price * units
            new_capacity[i] = 0
        }
        total -= cost
        if (total < 0) break
        res[0] += 1
    }
    return res
}

let iter = (inp) => {
    console.log(inp.length)
    let res = []
    for (let i = 0; i < inp.length; i++) {
        let call = inp[i]
        console.log(hard_sayed(30, call)[0])
    }
    return res
}

var wordBreak = function (s, wordDict) {
    let res = []
    let words = []
    const backtrack = function (idx) {
        if (idx === s.length) return 1 && res.push(words.join(' '))
        for (let i = idx; i < s.length; i++) {
            let word = s.substring(idx, i + 1)
            if (!wordDict.includes(word)) continue
            words.push(word)
            backtrack(i + 1)
            words.pop()
        }
    }
    backtrack(0)
    return res
};


var checkRecord = function (n) {
    const MOD = (10 ** 9) + 7
    const memo = {}
    const dp = (depth, A, L) => {
        let key = `${depth},${A},${L}`
        if (memo[key]) return memo[key]
        if (A === 2 || L === 3) return 0
        if (depth >= n) return 1
        let path = 0
        path += (A === 0 && dp(depth + 1, A + 1, 0)) +
            (L < 2 && dp(depth + 1, A, L + 1)) +
            dp(depth + 1, A, 0)

        memo[key] = path % MOD
        return memo[key]
    }
    return dp(0, 0, 0)
};

var checkRecord = function (n) {
    const MOD = (10 ** 9) + 7;
    let memo = new Array(n + 1).fill(0).map(() => new Array(2).fill(0).map(() => new Array(3).fill(-1)));
    console.log(memo)
    const dp = (depth, A, L) => {
        if (depth === n) return 1;
        if (A >= 2 || L >= 3) return 0;
        if (memo[depth][A][L] !== -1) return memo[depth][A][L];
        let path = (A === 0 && dp(depth + 1, A + 1, 0) % MOD) +
            (L < 2 && dp(depth + 1, A, L + 1) % MOD) +
            dp(depth + 1, A, 0)
        memo[depth][A][L] = path % MOD;
        return path % MOD
    }
    return dp(0, 0, 0);
};


var equalSubstring = function (s, t, maxCost) {
    let l = 0, maxLen = 0, cost = 0
    for (let r = 0; r < s.length; r++) {
        cost += Math.abs(s[r].charCodeAt(0) - t[r].charCodeAt(0))
        while (cost > maxCost) {
            cost -= Math.abs(s[l].charCodeAt(0) - t[l].charCodeAt(0))
            l++
        }
        maxLen = Math.max(maxLen, r - l + 1)
    }
    return maxLen
};

// time complexity : O(N^3)
// var countTriplets = function (arr) {
//     let a = 0, b = 0
//     let res = 0
//     for (let i = 0; i < arr.length - 1; i++) {
//         a = 0
//         for (let j = i + 1; j < arr.length; j++) {
//             a ^= arr[j - 1]
//             b = 0
//             for (let k = j; k < arr.length; k++) {
//                 b ^= arr[k]
//                 if(a == b) res++
//             }
//         }
//     }
//     return res
// };


// time complexity : O(N^2)
var countTriplets = function (arr) {
    let res = 0
    for (let i = 0; i < arr.length - 1; i++) {
        let cur_xor = arr[i]
        for (let k = i + 1; k < arr.length; k++) {
            cur_xor ^= arr[k]
            if (cur_xor === 0) res += k - i
        }
    }
    return res
};

var singleNumber = function (nums) {
    // each dublicated number will cancel itself and the xorSum will be equall to first_unique ^ second_unique 
    let xorSum = nums.reduce((acc, cur) => acc ^ cur, 0)
    let a = 0
    let b = 0
    let isolationBit = xorSum & -xorSum
    // seperate nums array into two groups based on one bit from the result
    // so we can isolate the first_unique in group and the second on group
    for (let num of nums) {
        if (num & isolationBit) {
            a ^= num
        } else {
            b ^= num
        }
    }
    return [a, b]
};

var appendCharacters = function (s, t) {
    let tp = 0;
    for (let i = 0; i < s.length; i++) tp += s[i] === t[tp];
    return t.length - tp;
};

var longestPalindrome = function (s) {
    let occ = new Map(), odd = 0, res = 0
    for (let char of s) {
        occ.set(char, (occ.get(char) || 0) + 1)
    }
    for (let [key, value] of occ) {
        if (value % 2 !== 1) odd = 1
        res += value % 2 === 0 ? value : value - 1
    }
    return res + odd
};

var relativeSortArray = function (arr1, arr2) {
    let freq = new Map(), res = [], rem = []
    for (let i = 0; i < arr1.length; i++) {
        if (!arr2.includes(arr1[i])) rem.push(arr1[i])
        freq.set(arr1[i], (freq.get(arr1[i]) || 0) + 1)
    }
    for (let i = 0; i < arr2.length; i++) {
        let repeat = freq.get(arr2[i])
        while (repeat--) {
            res.push(arr2[i])
        }
    }
    rem.sort((a, b) => a - b)
    return [...res, ...rem]
}

var heightChecker = function (heights) {
    let sorted = heights.toSorted((a, b) => a - b), res = 0
    heights.reduce((_, __, idx) => {
        res += heights[idx] !== sorted[idx]
    }, 0)
    return res
}

var subarrayDivByK = function (nums, k) {
    let prefixSum = 0, hash = new Map(), res = 0
    hash.set(0, 1)
    for (let n of nums) {
        prefixSum += n
        let remainder = (prefixSum % k + k) % k
        res += hash.get(remainder) ? hash.get(remainder) : 0
        hash.set(remainder, (hash.get(remainder) || 0) + 1)
    }
    return res
}


var solveNQueens = function (n) {
    const cols = new Set()
    const diagonals = new Set()
    const antiDiagonals = new Set()
    const res = []
    const board = Array(n).fill().map(() => Array(n).fill("."))
    const backtrack = (r) => {
        if (r === n) {
            let b = []
            for (let r of board) {
                b.push(r.join(""))
            }
            res.push(b)
            return
        }
        for (let col = 0; col < n; col++) {
            if (cols.has(col) || diagonals.has(r - col) || antiDiagonals.has(r + col)) continue
            cols.add(col)
            diagonals.add(r - col)
            antiDiagonals.add(r + col)
            board[r][col] = "Q"
            backtrack(r + 1)
            // console.log(board)
            board[r][col] = "."
            cols.delete(col)
            diagonals.delete(r - col)
            antiDiagonals.delete(r + col)
        }
    }
    backtrack(0)
    return res
};

var sortColors = function (nums) {
    let twoPtr = nums.length - 1, zeroPtr = 0
    let i = 0
    while (i < nums.length) {
        while (nums[twoPtr] === 2) twoPtr--
        while (nums[zeroPtr] === 0) zeroPtr++
        if (nums[i] === 2 && i < twoPtr) [nums[twoPtr], nums[i]] = [nums[i], nums[twoPtr]]
        else if (nums[i] === 0 && i > zeroPtr) [nums[zeroPtr], nums[i]] = [nums[i], nums[zeroPtr]]
        else i++
    }
    return nums
}

var minMovesToSeat = function (seats, students) {
    let res = 0
    let max = Math.max(...seats, ...students)
    let seatsSorted = Array(max).fill(0)
    let studentsSorted = Array(max).fill(0)
    for (let i = 0; i < students.length; i++) {
        seatsSorted[seats[i] - 1] += 1
        studentsSorted[students[i] - 1] += 1
    }
    let seatsPtr = 0
    let i = 0

    while (i < studentsSorted.length) {
        while (studentsSorted[i] === 0) i++
        while (seatsSorted[seatsPtr] === 0) seatsPtr++
        let min = Math.min(studentsSorted[i], seatsSorted[seatsPtr])
        if (isNaN(min)) break
        res += min * Math.abs(i - seatsPtr)
        seatsSorted[seatsPtr] -= min
        studentsSorted[i] -= min
    }

    return res
};

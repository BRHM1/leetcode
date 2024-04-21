function TreeNode(val, left, right) {
    this.val = (val === undefined ? 0 : val)
    this.left = (left === undefined ? null : left)
    this.right = (right === undefined ? null : right)
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
    parent = []
    constructor(parent){
        this.parent = parent
    }
    find(node){ // returns the representive of the group for a specific node
        if(this.parent[node] !== node) {
            return this.find(parent[node])
        }else {
            return node
        }
    }
    union(group1 , group2) {
       return this.parent[group1] = find(group2)
    }
}

const validPath = (n , edges , source , destination) => {
    const nodes = new Map(Array.from(Array(n).keys()).map(i => [i , i])) // make each node is the parent(representive) of itself
    const find = (node) => { // find's the representive of the input node 
        let root = node
        while(root !== nodes.get(root)){
            root = nodes.get(root)
        }
        return root
    }
    const union = (firstNode , secondNode) => { // group's two trees together under one root 
        let firstNodeRoot = find(firstNode)
        let secondNodeRoot = find(secondNode)
        nodes.set(firstNodeRoot , secondNodeRoot)
    }
    const isConnected = (first , second) => {
        return find(first) === find(second)
    }
    for(let [src , des] of edges) {
        union(src , des)
    }
    return isConnected(source , destination)
}
console.log(validPath(3 , [[0,1],[1,2],[2,0]] , 0 , 2))


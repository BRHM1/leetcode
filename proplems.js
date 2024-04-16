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

const tree = new TreeNode(4, new TreeNode(9, new TreeNode(5), new TreeNode(1)), new TreeNode(0))
// const tree = new TreeNode(1 , new TreeNode(2) , new TreeNode(3) )
console.log(sumNumbers(tree))


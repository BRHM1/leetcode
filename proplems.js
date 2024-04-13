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

var largestRectangleArea = function(heights) {
    let stack = [] // [starting index , value]
    let max = 0
    for(let i = 0; i < heights.length; i++){
        let start = i
        while(stack.length && stack.at(-1)[1] > heights[i]){
            let [idx , value] = stack.pop()
            let area = (i - idx) * value
            max = Math.max(max , area)
            start = idx
        }
        stack.push([start , heights[i]])
    }
    // the remaining items in the stack made it tell the end of the stack so the area = (heights.length - idx) * value
    while(stack.length){
        let [idx , value] = stack.pop()
        let area = (heights.length - idx) * value
        max = Math.max(area , max)
    }
    return max
};
console.log(largestRectangleArea([2,1,5,6,2,3])) // 10
console.log(largestRectangleArea([2,4])) // 4

var maximalRectangle = function (matrix) {
    // approach : make two grids one for the max width ending at that point second one for the height
    let ROWS = matrix.length, COLS = matrix[0].length
    const width = Array(ROWS).fill().map(() => Array(COLS).fill(0))
    const height = Array(ROWS).fill().map(() => Array(COLS).fill(0))
    let max = 0
    function getTheFirstDecreasingRow(num){
        // go and binary search in the width matrix to get the first uncompleted row
    }
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            // if the height goes above 1 that means there is a factor width (min)
            // logic for height calc
            if (r - 1 >= 0) {
                height[r][c] += matrix[r][c] === "1" ? (height[r - 1][c] || 0) + 1 : 0
            } else {
                height[r][c] += +(matrix[r][c])
            }
            // logic for width calc
            width[r][c] += matrix[r][c] === "1" ? (width[r][c - 1] || 0) + 1 : 0
            max = Math.max(max, width[r][c], height[r][c])
            if (height[r][c] > 1) {
                // width[r][c] = Math.min(width[r][c], width[r - 1][c])
                for(let i = 1; i < height[r][c]; i++){
                    max = Math.max(max, Math.min(width[r][c], width[r - i][c]) * height[r - i][c])
                }
            } else {
                max = Math.max(max, width[r][c] * height[r][c])
            }
        }
    }
    console.log("w", width)
    console.log("h", height)
    return max
};
// console.log(maximalRectangle([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]])) // 6
// console.log(maximalRectangle([["0" ,"0","1","1"],["0","1","1","1"],["0","1" , "1" , "1"]])) // 6
// console.log(maximalRectangle([["1", "0", "1"], ["1", "1", "0"]])) // 2
// console.log(maximalRectangle([["0"]])) // 0
// console.log(maximalRectangle([["1"]])) // 1
// console.log(maximalRectangle([["0","0","1"],["1","1","1"]])) // 3
// console.log(maximalRectangle([["0", "0", "1", "1"], ["0", "1", "1", "1"], ["0", "1", "1", "1"], ["1", "1", "1", "1"]])) // 9

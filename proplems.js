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

var maximalRectangle = function (matrix) {
    // approach : combine each row with the row on top of it to create histogram then calculate the max area in that histogram
    // if there is a zero it makes the total col = 0
    let max = 0
    let histogram_arr = Array(matrix.length).fill().map(() => Array(matrix[0].length))
    for(let r = 0; r < matrix.length; r++){
        for(let c = 0; c < matrix[0].length; c++){
            if(r === 0){
                histogram_arr[r][c] = +matrix[r][c]
            }else if(matrix[r][c] === "0"){
                histogram_arr[r][c] = 0
            }else {
                histogram_arr[r][c] = +histogram_arr[r - 1][c] + 1
            }
        }
    }
    for(let histogram of histogram_arr){
        max = Math.max(max , largestRectangleArea(histogram))
    }
    return max
};

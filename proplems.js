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
console.log(removeKdigits("10200", 1)) // 200
console.log(removeKdigits("1432219", 3)) // 1219
console.log(removeKdigits("9", 1)) // 0
console.log(removeKdigits("1112", 1)) // 111
console.log(removeKdigits("43214321", 4)) // 1321
console.log(removeKdigits("10001", 4)) // 0
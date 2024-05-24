var maxScoreWords = function (words, letters, score) {
    // backtrack each subset i can take and know the global count at one moment 
    // if i exceed the global count it's no more valid path
    let res = 0
    let count = new Map()
    for (let letter of letters) {
        count.set(letter, (count.get(letter) || 0) + 1)
    }
    const backtrack = (idx, current_res) => {
        // determine if this is a valid path if not return
        res = Math.max(res, current_res)
        if (idx >= words.length) return 
        let temp = 0
        let temp_count = new Map()
        for (let letter of words[idx]) {
            temp += score[letter.charCodeAt(0) - 97]
            if(count.has(letter)) count.set(letter, count.get(letter) - 1);
            if(count.has(letter)) temp_count.set(letter, (temp_count.get(letter) || 0) + 1);
            console.log(count)
            // the path is no more valid 
            if (count.get(letter) < 0) {
                for (let [letter, value] of temp_count) {
                    count.set(letter, count.get(letter) + value)
                }
                return
            }
        }

        // add to current result
        current_res += temp
        backtrack(idx + 1, current_res, true)
        // remove the added value to current result
        current_res -= temp
        // add the decremented count to the path
        for (let [letter, value] of temp_count) {
            count.set(letter, count.get(letter) + value)
        }
        backtrack(idx + 1, current_res, false)
    }
    backtrack(0, 0, true)
    backtrack(0, 0, false)
    return res
};
    let squareRoot = Math.floor(Math.sqrt(c))
    let l = 0, r = squareRoot
    while (l <= r) {
        let mid = Math.floor((l + r) / 2)
        let res = l ** 2 + r ** 2
        if (res === c) {
            return true
        } else if (res > c) {
            r = mid - 1
        } else {
            l = mid + 1
        }
    }
    return false
};
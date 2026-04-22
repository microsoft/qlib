/**
 * 保留 object 中的部分内容
 * @param obj
 * @param keys
 */
export function pick(obj, keys) {
    var r = {};
    keys.forEach(function (key) {
        r[key] = obj[key];
    });
    return r;
}
//# sourceMappingURL=pick.js.map
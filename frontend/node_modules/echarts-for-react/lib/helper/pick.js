"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.pick = void 0;
/**
 * 保留 object 中的部分内容
 * @param obj
 * @param keys
 */
function pick(obj, keys) {
    var r = {};
    keys.forEach(function (key) {
        r[key] = obj[key];
    });
    return r;
}
exports.pick = pick;
//# sourceMappingURL=pick.js.map
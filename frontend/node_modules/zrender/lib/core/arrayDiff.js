function diff(oldArr, newArr, equals) {
    if (!equals) {
        equals = function (a, b) {
            return a === b;
        };
    }
    oldArr = oldArr.slice();
    newArr = newArr.slice();
    var newLen = newArr.length;
    var oldLen = oldArr.length;
    var editLength = 1;
    var maxEditLength = newLen + oldLen;
    var bestPath = [{ newPos: -1, components: [] }];
    var oldPos = extractCommon(bestPath[0], newArr, oldArr, 0, equals);
    if (!oldLen
        || !newLen
        || (bestPath[0].newPos + 1 >= newLen && oldPos + 1 >= oldLen)) {
        var indices = [];
        var allCleared = !newLen && oldLen > 0;
        var allCreated = !oldLen && newLen > 0;
        for (var i = 0; i < (allCleared ? oldArr : newArr).length; i++) {
            indices.push(i);
        }
        return [{
                indices: indices,
                count: indices.length,
                added: allCreated,
                removed: allCleared
            }];
    }
    function execEditLength() {
        for (var diagonalPath = -1 * editLength; diagonalPath <= editLength; diagonalPath += 2) {
            var basePath;
            var addPath = bestPath[diagonalPath - 1];
            var removePath = bestPath[diagonalPath + 1];
            var oldPos = (removePath ? removePath.newPos : 0) - diagonalPath;
            if (addPath) {
                bestPath[diagonalPath - 1] = undefined;
            }
            var canAdd = addPath && addPath.newPos + 1 < newLen;
            var canRemove = removePath && 0 <= oldPos && oldPos < oldLen;
            if (!canAdd && !canRemove) {
                bestPath[diagonalPath] = undefined;
                continue;
            }
            if (!canAdd || (canRemove && addPath.newPos < removePath.newPos)) {
                basePath = clonePath(removePath);
                pushComponent(basePath.components, false, true);
            }
            else {
                basePath = addPath;
                basePath.newPos++;
                pushComponent(basePath.components, true, false);
            }
            oldPos = extractCommon(basePath, newArr, oldArr, diagonalPath, equals);
            if (basePath.newPos + 1 >= newLen && oldPos + 1 >= oldLen) {
                return buildValues(basePath.components);
            }
            else {
                bestPath[diagonalPath] = basePath;
            }
        }
        editLength++;
    }
    while (editLength <= maxEditLength) {
        var ret = execEditLength();
        if (ret) {
            return ret;
        }
    }
}
function extractCommon(basePath, newArr, oldArr, diagonalPath, equals) {
    var newLen = newArr.length;
    var oldLen = oldArr.length;
    var newPos = basePath.newPos;
    var oldPos = newPos - diagonalPath;
    var commonCount = 0;
    while (newPos + 1 < newLen && oldPos + 1 < oldLen && equals(newArr[newPos + 1], oldArr[oldPos + 1])) {
        newPos++;
        oldPos++;
        commonCount++;
    }
    if (commonCount) {
        basePath.components.push({
            count: commonCount,
            added: false,
            removed: false,
            indices: []
        });
    }
    basePath.newPos = newPos;
    return oldPos;
}
function pushComponent(components, added, removed) {
    var last = components[components.length - 1];
    if (last && last.added === added && last.removed === removed) {
        components[components.length - 1] = {
            count: last.count + 1,
            added: added,
            removed: removed,
            indices: []
        };
    }
    else {
        components.push({
            count: 1,
            added: added,
            removed: removed,
            indices: []
        });
    }
}
function buildValues(components) {
    var componentPos = 0;
    var componentLen = components.length;
    var newPos = 0;
    var oldPos = 0;
    for (; componentPos < componentLen; componentPos++) {
        var component = components[componentPos];
        if (!component.removed) {
            var indices = [];
            for (var i = newPos; i < newPos + component.count; i++) {
                indices.push(i);
            }
            component.indices = indices;
            newPos += component.count;
            if (!component.added) {
                oldPos += component.count;
            }
        }
        else {
            for (var i = oldPos; i < oldPos + component.count; i++) {
                component.indices.push(i);
            }
            oldPos += component.count;
        }
    }
    return components;
}
function clonePath(path) {
    return { newPos: path.newPos, components: path.components.slice(0) };
}
export default function arrayDiff(oldArr, newArr, equal) {
    return diff(oldArr, newArr, equal);
}

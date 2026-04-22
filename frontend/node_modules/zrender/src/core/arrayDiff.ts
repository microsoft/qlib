// Myers' Diff Algorithm
// Modified from https://github.com/kpdecker/jsdiff/blob/master/src/diff/base.js
type EqualFunc<T> = (a: T, b: T) => boolean;

type DiffComponent = {
    count: number
    added: boolean
    removed: boolean,
    indices: number[]
}

type DiffPath = {
    components: DiffComponent[],
    newPos: number
}

// Using O(ND) algorithm
// TODO: Optimize when diff is large.
function diff<T>(oldArr: T[], newArr: T[], equals: EqualFunc<T>): DiffComponent[] {
    if (!equals) {
        equals = function (a, b) {
            return a === b;
        };
    }

    oldArr = oldArr.slice();
    newArr = newArr.slice();
    // Allow subclasses to massage the input prior to running
    var newLen = newArr.length;
    var oldLen = oldArr.length;
    var editLength = 1;
    var maxEditLength = newLen + oldLen;
    var bestPath: DiffPath[] = [{ newPos: -1, components: [] }];

    // Seed editLength = 0, i.e. the content starts with the same values
    var oldPos = extractCommon<T>(bestPath[0], newArr, oldArr, 0, equals);
    if (!oldLen // All new created
        || !newLen // Clear
        || (bestPath[0].newPos + 1 >= newLen && oldPos + 1 >= oldLen)) {
        var indices = [];
        var allCleared = !newLen && oldLen > 0;
        var allCreated = !oldLen && newLen > 0;
        for (let i = 0; i < (allCleared ? oldArr : newArr).length; i++) {
            indices.push(i);
        }
        // Identity per the equality and tokenizer
        return [{
            indices: indices,
            count: indices.length,
            added: allCreated,
            removed: allCleared
        }];
    }

    // Main worker method. checks all permutations of a given edit length for acceptance.
    function execEditLength() {
        for (let diagonalPath = -1 * editLength; diagonalPath <= editLength; diagonalPath += 2) {
            var basePath;
            var addPath = bestPath[diagonalPath - 1];
            var removePath = bestPath[diagonalPath + 1];
            var oldPos = (removePath ? removePath.newPos : 0) - diagonalPath;
            if (addPath) {
                // No one else is going to attempt to use this value, clear it
                bestPath[diagonalPath - 1] = undefined;
            }

            var canAdd = addPath && addPath.newPos + 1 < newLen;
            var canRemove = removePath && 0 <= oldPos && oldPos < oldLen;
            if (!canAdd && !canRemove) {
                // If this path is a terminal then prune
                bestPath[diagonalPath] = undefined;
                continue;
            }

            // Select the diagonal that we want to branch from. We select the prior
            // path whose position in the new string is the farthest from the origin
            // and does not pass the bounds of the diff graph
            if (!canAdd || (canRemove && addPath.newPos < removePath.newPos)) {
                basePath = clonePath(removePath);
                pushComponent(basePath.components, false, true);
            }
            else {
                basePath = addPath;   // No need to clone, we've pulled it from the list
                basePath.newPos++;
                pushComponent(basePath.components, true, false);
            }

            oldPos = extractCommon<T>(basePath, newArr, oldArr, diagonalPath, equals);

            // If we have hit the end of both strings, then we are done
            if (basePath.newPos + 1 >= newLen && oldPos + 1 >= oldLen) {
                return buildValues(basePath.components);
            }
            else {
                // Otherwise track this path as a potential candidate and continue.
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

function extractCommon<T>(basePath: DiffPath, newArr: T[], oldArr: T[], diagonalPath: number, equals: EqualFunc<T>) {
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

function pushComponent(components: DiffComponent[], added: boolean, removed: boolean) {
    var last = components[components.length - 1];
    if (last && last.added === added && last.removed === removed) {
        // We need to clone here as the component clone operation is just
        // as shallow array clone
        components[components.length - 1] = {
            count: last.count + 1,
            added,
            removed,
            indices: []
        };
    }
    else {
        components.push({
            count: 1,
            added,
            removed,
            indices: []
        });
    }
}

function buildValues(components: DiffComponent[]) {
    var componentPos = 0;
    var componentLen = components.length;
    var newPos = 0;
    var oldPos = 0;

    for (; componentPos < componentLen; componentPos++) {
        var component = components[componentPos];
        if (!component.removed) {
            var indices = [];
            for (let i = newPos; i < newPos + component.count; i++) {
                indices.push(i);
            }
            component.indices = indices;
            newPos += component.count;
            // Common case
            if (!component.added) {
                oldPos += component.count;
            }
        }
        else {
            for (let i = oldPos; i < oldPos + component.count; i++) {
                component.indices.push(i);
            }
            oldPos += component.count;
        }
    }

    return components;
}

function clonePath(path: DiffPath) {
    return { newPos: path.newPos, components: path.components.slice(0) };
}

export default function arrayDiff<T>(oldArr: T[], newArr: T[], equal?: EqualFunc<T>): DiffComponent[] {
    return diff(oldArr, newArr, equal);
}
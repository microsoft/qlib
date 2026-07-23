/**
 * The algoritm is learnt from
 * https://franklinta.com/2014/09/08/computing-css-matrix3d-transforms/
 * And we made some optimization for matrix inversion.
 * Other similar approaches:
 * "cv::getPerspectiveTransform", "Direct Linear Transformation".
 */

const LN2 = Math.log(2);

function determinant(
    rows: number[][],
    rank: number,
    rowStart: number,
    rowMask: number,
    colMask: number,
    detCache: {[key: string]: number}
) {
    const cacheKey = rowMask + '-' + colMask;
    const fullRank = rows.length;

    if (detCache.hasOwnProperty(cacheKey)) {
        return detCache[cacheKey];
    }

    if (rank === 1) {
        // In this case the colMask must be like: `11101111`. We can find the place of `0`.
        const colStart = Math.round(Math.log(((1 << fullRank) - 1) & ~colMask) / LN2);
        return rows[rowStart][colStart];
    }

    const subRowMask = rowMask | (1 << rowStart);
    let subRowStart = rowStart + 1;
    while (rowMask & (1 << subRowStart)) {
        subRowStart++;
    }

    let sum = 0;
    for (let j = 0, colLocalIdx = 0; j < fullRank; j++) {
        const colTag = 1 << j;
        if (!(colTag & colMask)) {
            sum += (colLocalIdx % 2 ? -1 : 1) * rows[rowStart][j]
                // det(subMatrix(0, j))
                * determinant(rows, rank - 1, subRowStart, subRowMask, colMask | colTag, detCache);
            colLocalIdx++;
        }
    }

    detCache[cacheKey] = sum;

    return sum;
}

/**
 * Usage:
 * ```js
 * const transformer = buildTransformer(
 *     [10, 44, 100, 44, 100, 300, 10, 300],
 *     [50, 54, 130, 14, 140, 330, 14, 220]
 * );
 * const out = [];
 * transformer && transformer([11, 33], out);
 * ```
 *
 * Notice: `buildTransformer` may take more than 10ms in some Android device.
 *
 * @param src source four points, [x0, y0, x1, y1, x2, y2, x3, y3]
 * @param dest destination four points, [x0, y0, x1, y1, x2, y2, x3, y3]
 * @return transformer If fail, return null/undefined.
 */
export function buildTransformer(src: number[], dest: number[]) {
    const mA = [
        [src[0], src[1], 1, 0, 0, 0, -dest[0] * src[0], -dest[0] * src[1]],
        [0, 0, 0, src[0], src[1], 1, -dest[1] * src[0], -dest[1] * src[1]],
        [src[2], src[3], 1, 0, 0, 0, -dest[2] * src[2], -dest[2] * src[3]],
        [0, 0, 0, src[2], src[3], 1, -dest[3] * src[2], -dest[3] * src[3]],
        [src[4], src[5], 1, 0, 0, 0, -dest[4] * src[4], -dest[4] * src[5]],
        [0, 0, 0, src[4], src[5], 1, -dest[5] * src[4], -dest[5] * src[5]],
        [src[6], src[7], 1, 0, 0, 0, -dest[6] * src[6], -dest[6] * src[7]],
        [0, 0, 0, src[6], src[7], 1, -dest[7] * src[6], -dest[7] * src[7]]
    ];

    const detCache = {};
    const det = determinant(mA, 8, 0, 0, 0, detCache);
    if (det === 0) {
        // can not make transformer when and only when
        // any three of the markers are collinear.
        return;
    }

    // `invert(mA) * dest`, that is, `adj(mA) / det * dest`.
    const vh: number[] = [];
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            vh[j] == null && (vh[j] = 0);
            vh[j] += ((i + j) % 2 ? -1 : 1)
                // det(subMatrix(i, j))
                * determinant(mA, 7, i === 0 ? 1 : 0, 1 << i, 1 << j, detCache)
                / det * dest[i];
        }
    }

    return function (out: number[], srcPointX: number, srcPointY: number) {
        const pk = srcPointX * vh[6] + srcPointY * vh[7] + 1;
        out[0] = (srcPointX * vh[0] + srcPointY * vh[1] + vh[2]) / pk;
        out[1] = (srcPointX * vh[3] + srcPointY * vh[4] + vh[5]) / pk;
    };
}

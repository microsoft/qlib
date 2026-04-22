/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

import Point, { PointLike } from './Point';
import BoundingRect from './BoundingRect';
import { MatrixArray } from './matrix';

const extent = [0, 0];
const extent2 = [0, 0];

const minTv = new Point();
const maxTv = new Point();

class OrientedBoundingRect {

    // lt, rt, rb, lb
    private _corners: Point[] = [];

    private _axes: Point[] = [];

    private _origin: number[] = [0, 0];

    constructor(rect?: BoundingRect, transform?: MatrixArray) {
        for (let i = 0; i < 4; i++) {
            this._corners[i] = new Point();
        }
        for (let i = 0; i < 2; i++) {
            this._axes[i] = new Point();
        }

        if (rect) {
            this.fromBoundingRect(rect, transform);
        }
    }

    fromBoundingRect(rect: BoundingRect, transform?: MatrixArray) {
        const corners = this._corners;
        const axes = this._axes;
        const x = rect.x;
        const y = rect.y;
        const x2 = x + rect.width;
        const y2 = y + rect.height;
        corners[0].set(x, y);
        corners[1].set(x2, y);
        corners[2].set(x2, y2);
        corners[3].set(x, y2);

        if (transform) {
            for (let i = 0; i < 4; i++) {
                corners[i].transform(transform);
            }
        }

        // Calculate axes
        Point.sub(axes[0], corners[1], corners[0]);
        Point.sub(axes[1], corners[3], corners[0]);
        axes[0].normalize();
        axes[1].normalize();

        // Calculate projected origin
        for (let i = 0; i < 2; i++) {
            this._origin[i] = axes[i].dot(corners[0]);
        }
    }

    /**
     * If intersect with another OBB
     * @param other Bounding rect to be intersected with
     * @param mtv Calculated .
     *  If it's not overlapped. it means needs to move given rect with Maximum Translation Vector to be overlapped.
     *  Else it means needs to move given rect with Minimum Translation Vector to be not overlapped.
     */
    intersect(other: OrientedBoundingRect, mtv?: PointLike): boolean {
        // OBB collision with SAT method

        let overlapped = true;
        const noMtv = !mtv;
        minTv.set(Infinity, Infinity);
        maxTv.set(0, 0);
        // Check two axes for both two obb.
        if (!this._intersectCheckOneSide(this, other, minTv, maxTv, noMtv, 1)) {
            overlapped = false;
            if (noMtv) {
                // Early return if no need to calculate mtv
                return overlapped;
            }
        }
        if (!this._intersectCheckOneSide(other, this, minTv, maxTv, noMtv, -1)) {
            overlapped = false;
            if (noMtv) {
                return overlapped;
            }
        }

        if (!noMtv) {
            Point.copy(mtv, overlapped ? minTv : maxTv);
        }

        return overlapped;
    }


    private _intersectCheckOneSide(
        self: OrientedBoundingRect,
        other: OrientedBoundingRect,
        minTv: Point,
        maxTv: Point,
        noMtv: boolean,
        inverse: 1 | -1
    ): boolean {
        let overlapped = true;
        for (let i = 0; i < 2; i++) {
            const axis = this._axes[i];
            this._getProjMinMaxOnAxis(i, self._corners, extent);
            this._getProjMinMaxOnAxis(i, other._corners, extent2);

            // Not overlap on the any axis.
            if (extent[1] < extent2[0] || extent[0] > extent2[1]) {
                overlapped = false;
                if (noMtv) {
                    return overlapped;
                }
                const dist0 = Math.abs(extent2[0] - extent[1]);
                const dist1 = Math.abs(extent[0] - extent2[1]);

                // Find longest distance of all axes.
                if (Math.min(dist0, dist1) > maxTv.len()) {
                    if (dist0 < dist1) {
                        Point.scale(maxTv, axis, -dist0 * inverse);
                    }
                    else {
                        Point.scale(maxTv, axis, dist1 * inverse);
                    }
                }
            }
            else if (minTv) {
                const dist0 = Math.abs(extent2[0] - extent[1]);
                const dist1 = Math.abs(extent[0] - extent2[1]);

                if (Math.min(dist0, dist1) < minTv.len()) {
                    if (dist0 < dist1) {
                        Point.scale(minTv, axis, dist0 * inverse);
                    }
                    else {
                        Point.scale(minTv, axis, -dist1 * inverse);
                    }
                }
            }
        }
        return overlapped;
    }

    private _getProjMinMaxOnAxis(dim: number, corners: Point[], out: number[]) {
        const axis = this._axes[dim];
        const origin = this._origin;
        const proj = corners[0].dot(axis) + origin[dim];
        let min = proj;
        let max = proj;

        for (let i = 1; i < corners.length; i++) {
            const proj = corners[i].dot(axis) + origin[dim];
            min = Math.min(proj, min);
            max = Math.max(proj, max);
        }

        out[0] = min;
        out[1] = max;
    }
}

export default OrientedBoundingRect;
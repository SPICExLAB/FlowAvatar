//using System;
//using UnityEngine;

///// <summary>
///// Utility class for coordinate transformations and rotation conversions
///// between Unity and various coordinate systems
///// </summary>
//public static class UnityTransformUtils
//{
//    #region Rotation Conversions

//    /// <summary>
//    /// Converts quaternion to axis-angle representation
//    /// </summary>
//    public static Vector3 QuaternionToAxisAngle(Quaternion q)
//    {
//        // Ensure quaternion is normalized
//        q.Normalize();

//        // Calculate rotation angle
//        float angle = 2.0f * Mathf.Acos(Mathf.Clamp(q.w, -1.0f, 1.0f));

//        // If angle is very small, return zero vector to avoid division by zero
//        if (Mathf.Approximately(angle, 0f))
//            return Vector3.zero;

//        // Calculate axis
//        float s = Mathf.Sqrt(1.0f - q.w * q.w);
//        if (s < 0.001f) // Handle numerical instability for very small angles
//        {
//            return new Vector3(q.x, q.y, q.z);
//        }
//        else
//        {
//            // Return axis * angle
//            float invs = 1.0f / s;
//            return new Vector3(q.x * invs, q.y * invs, q.z * invs) * angle;
//        }
//    }

//    /// <summary>
//    /// Converts axis-angle representation to quaternion
//    /// </summary>
//    public static Quaternion AxisAngleToQuaternion(Vector3 axisAngle)
//    {
//        float angle = axisAngle.magnitude;

//        if (angle < 0.0001f)
//            return Quaternion.identity;

//        Vector3 axis = axisAngle.normalized;
//        return Quaternion.AngleAxis(angle * Mathf.Rad2Deg, axis);
//    }

//    /// <summary>
//    /// Converts rotation matrix (3x3) to 6D representation (first two columns)
//    /// </summary>
//    public static Vector3[] MatrixToSixD(Matrix4x4 rotMatrix)
//    {
//        Vector3[] sixD = new Vector3[2];

//        // First column (3 values)
//        sixD[0] = new Vector3(
//            rotMatrix.m00,
//            rotMatrix.m10,
//            rotMatrix.m20
//        );

//        // Second column (3 values)
//        sixD[1] = new Vector3(
//            rotMatrix.m01,
//            rotMatrix.m11,
//            rotMatrix.m21
//        );

//        return sixD;
//    }

//    /// <summary>
//    /// Converts 6D representation (two basis vectors) to rotation matrix
//    /// Matches Python sixd2matrot function
//    /// </summary>
//    public static Matrix4x4 SixDToMatrix(Vector3 col1, Vector3 col2)
//    {
//        // Normalize first column
//        Vector3 b1 = col1.normalized;

//        // Calculate projection of col2 onto b1
//        float dot = Vector3.Dot(b1, col2);
//        Vector3 projection = dot * b1;

//        // Orthogonalize and normalize second column
//        Vector3 b2 = (col2 - projection).normalized;

//        // Get third column as cross product
//        Vector3 b3 = Vector3.Cross(b1, b2);

//        Matrix4x4 mat = Matrix4x4.identity;

//        // First column
//        mat.m00 = b1.x; mat.m10 = b1.y; mat.m20 = b1.z;

//        // Second column
//        mat.m01 = b2.x; mat.m11 = b2.y; mat.m21 = b2.z;

//        // Third column
//        mat.m02 = b3.x; mat.m12 = b3.y; mat.m22 = b3.z;

//        return mat;
//    }

//    /// <summary>
//    /// Converts 6D rotation to quaternion
//    /// </summary>
//    public static Quaternion SixDToQuaternion(Vector3 col1, Vector3 col2)
//    {
//        Matrix4x4 matrix = SixDToMatrix(col1, col2);
//        return QuaternionFromMatrix(matrix);
//    }

//    /// <summary>
//    /// Extracts a Quaternion from a rotation matrix
//    /// </summary>
//    public static Quaternion QuaternionFromMatrix(Matrix4x4 m)
//    {
//        float trace = m.m00 + m.m11 + m.m22;
//        Quaternion q = new Quaternion();

//        if (trace > 0)
//        {
//            float s = 0.5f / Mathf.Sqrt(trace + 1.0f);
//            q.w = 0.25f / s;
//            q.x = (m.m21 - m.m12) * s;
//            q.y = (m.m02 - m.m20) * s;
//            q.z = (m.m10 - m.m01) * s;
//        }
//        else
//        {
//            if (m.m00 > m.m11 && m.m00 > m.m22)
//            {
//                float s = 2.0f * Mathf.Sqrt(1.0f + m.m00 - m.m11 - m.m22);
//                q.w = (m.m21 - m.m12) / s;
//                q.x = 0.25f * s;
//                q.y = (m.m01 + m.m10) / s;
//                q.z = (m.m02 + m.m20) / s;
//            }
//            else if (m.m11 > m.m22)
//            {
//                float s = 2.0f * Mathf.Sqrt(1.0f + m.m11 - m.m00 - m.m22);
//                q.w = (m.m02 - m.m20) / s;
//                q.x = (m.m01 + m.m10) / s;
//                q.y = 0.25f * s;
//                q.z = (m.m12 + m.m21) / s;
//            }
//            else
//            {
//                float s = 2.0f * Mathf.Sqrt(1.0f + m.m22 - m.m00 - m.m11);
//                q.w = (m.m10 - m.m01) / s;
//                q.x = (m.m02 + m.m20) / s;
//                q.y = (m.m12 + m.m21) / s;
//                q.z = 0.25f * s;
//            }
//        }

//        return q.normalized;
//    }

//    /// <summary>
//    /// Converts from 6D rotation to axis-angle representation
//    /// </summary>
//    public static Vector3 SixDToAxisAngle(Vector3 col1, Vector3 col2)
//    {
//        Quaternion rotation = SixDToQuaternion(col1, col2);
//        return QuaternionToAxisAngle(rotation);
//    }

//    #endregion

//    #region Coordinate Transformations

//    /// <summary>
//    /// Converts from Unity coordinates to the AMASS/ML model's coordinate system
//    /// </summary>
//    public static void TransformCoordinatesToAMASS(
//        Vector3[] positions,
//        Quaternion[] rotations,
//        out Vector3[] transformedPositions,
//        out Quaternion[] transformedRotations)
//    {
//        // Initialize coordinate transform matrices
//        Matrix4x4 coordTransform = Matrix4x4.Rotate(Quaternion.Euler(90, 0, 0));

//        // Create left hand offset matrix (180 degrees around Z)
//        Matrix4x4 handLeftOffset = Matrix4x4.Rotate(Quaternion.Euler(180, 0, 0));

//        Matrix4x4 handRightOffset = Matrix4x4.identity;

//        transformedPositions = new Vector3[positions.Length];
//        transformedRotations = new Quaternion[rotations.Length];

//        for (int i = 0; i < positions.Length; i++)
//        {
//            // 1. Negate X for handedness conversion
//            Vector3 flippedPosition = new Vector3(-positions[i].x, positions[i].y, positions[i].z);

//            // 2. Apply coordinate transform (90-degree X rotation)
//            transformedPositions[i] = coordTransform.MultiplyPoint3x4(flippedPosition);

//            // 3. Handle quaternion handedness by negating Y and Z
//            Quaternion flippedRotation = new Quaternion(
//                rotations[i].x,
//                -rotations[i].y,
//                -rotations[i].z,
//                rotations[i].w
//            );

//            // 4. Apply hand-specific offsets and coordinate transform
//            Matrix4x4 finalRotMatrix;

//            if (i == 1) // Left hand
//            {
//                Matrix4x4 rotMatrix = Matrix4x4.Rotate(flippedRotation) * handLeftOffset;
//                finalRotMatrix = coordTransform * rotMatrix;
//            }
//            else if (i == 2) // Right hand
//            {
//                Matrix4x4 rotMatrix = Matrix4x4.Rotate(flippedRotation) * handRightOffset;
//                finalRotMatrix = coordTransform * rotMatrix;
//            }
//            else // Head
//            {
//                finalRotMatrix = coordTransform * Matrix4x4.Rotate(flippedRotation);
//            }

//            // Extract rotation from matrix
//            transformedRotations[i] = QuaternionFromMatrix(finalRotMatrix);
//        }
//    }

//    /// <summary>
//    /// Transform hand positions and rotations to head space
//    /// </summary>
//    public static void TransformToHeadSpace(
//        Vector3 headPosition,
//        Quaternion headRotation,
//        Vector3[] handPositions,
//        Quaternion[] handRotations,
//        out Vector3[] handLocalPositions,
//        out Quaternion[] handLocalRotations)
//    {
//        handLocalPositions = new Vector3[handPositions.Length];
//        handLocalRotations = new Quaternion[handRotations.Length];

//        // Calculate inverse head rotation
//        Quaternion headRotInv = Quaternion.Inverse(headRotation);
//        Matrix4x4 headRotMatrixInv = Matrix4x4.Rotate(headRotInv);

//        for (int i = 0; i < handPositions.Length; i++)
//        {
//            // Calculate position offset in world space
//            Vector3 posOffset = handPositions[i] - headPosition;

//            // Calculate hand position in head space using matrix multiplication
//            handLocalPositions[i] = headRotMatrixInv.MultiplyVector(posOffset);

//            // Calculate hand rotation in head space
//            handLocalRotations[i] = headRotInv * handRotations[i];
//        }
//    }

//    /// <summary>
//    /// Creates a 90D feature vector from tracking data (headspace version)
//    /// </summary>
//    public static float[] CreateHeadspaceFeatures(
//        Vector3[] globalPositions,
//        Vector3[] globalPosVelocities,
//        Vector3[] globalRotations6D,
//        Vector3[] globalRotVelocities6D,
//        Vector3[] handLocalPositions,
//        Vector3[] handLocalPosVelocities,
//        Vector3[] handLocalRotations6D,
//        Vector3[] handLocalRotVelocities6D)
//    {
//        // Feature order:
//        // Global 6D rotations [0:18]
//        // Rotation velocities [18:36]
//        // Global positions [36:45]
//        // Position velocities [45:54]
//        // Hand rotations in head space [54:66]
//        // Hand rotation velocities in head space [66:78]
//        // Hand positions in head space [78:84]
//        // Hand position velocities in head space [84:90]

//        const int FEATURE_SIZE = 90;
//        float[] features = new float[FEATURE_SIZE];
//        int offset = 0;

//        // Global 6D rotations [0:18]
//        for (int i = 0; i < globalRotations6D.Length; i++)
//        {
//            features[offset++] = globalRotations6D[i].x;
//            features[offset++] = globalRotations6D[i].y;
//            features[offset++] = globalRotations6D[i].z;
//        }

//        // Global rotation velocities [18:36]
//        for (int i = 0; i < globalRotVelocities6D.Length; i++)
//        {
//            features[offset++] = globalRotVelocities6D[i].x;
//            features[offset++] = globalRotVelocities6D[i].y;
//            features[offset++] = globalRotVelocities6D[i].z;
//        }

//        // Global positions [36:45]
//        for (int i = 0; i < globalPositions.Length; i++)
//        {
//            features[offset++] = globalPositions[i].x;
//            features[offset++] = globalPositions[i].y;
//            features[offset++] = globalPositions[i].z;
//        }

//        // Global position velocities [45:54]
//        for (int i = 0; i < globalPosVelocities.Length; i++)
//        {
//            features[offset++] = globalPosVelocities[i].x;
//            features[offset++] = globalPosVelocities[i].y;
//            features[offset++] = globalPosVelocities[i].z;
//        }

//        // Hand rotations in head space [54:66]
//        for (int i = 0; i < handLocalRotations6D.Length; i++)
//        {
//            features[offset++] = handLocalRotations6D[i].x;
//            features[offset++] = handLocalRotations6D[i].y;
//            features[offset++] = handLocalRotations6D[i].z;
//        }

//        // Hand rotation velocities in head space [66:78]
//        for (int i = 0; i < handLocalRotVelocities6D.Length; i++)
//        {
//            features[offset++] = handLocalRotVelocities6D[i].x;
//            features[offset++] = handLocalRotVelocities6D[i].y;
//            features[offset++] = handLocalRotVelocities6D[i].z;
//        }

//        // Hand positions in head space [78:84]
//        for (int i = 0; i < handLocalPositions.Length; i++)
//        {
//            features[offset++] = handLocalPositions[i].x;
//            features[offset++] = handLocalPositions[i].y;
//            features[offset++] = handLocalPositions[i].z;
//        }

//        // Hand position velocities in head space [84:90]
//        for (int i = 0; i < handLocalPosVelocities.Length; i++)
//        {
//            features[offset++] = handLocalPosVelocities[i].x;
//            features[offset++] = handLocalPosVelocities[i].y;
//            features[offset++] = handLocalPosVelocities[i].z;
//        }

//        return features;
//    }

//    /// <summary>
//    /// Converts from ML model's coordinate system to Unity coordinates
//    /// </summary>
//    public static void ConvertSMPLToUnity(Vector3[] poses, Vector3 headPosition, ref Vector3 translation)
//    {
//        // Create rotation matrix for coordinate transform (-90 degrees around X)
//        Quaternion coordTransform = Quaternion.Euler(-90, 0, 0);
//        Matrix4x4 rotMatrix = Matrix4x4.Rotate(coordTransform);

//        // 1. Transform translation using head position
//        Vector3 transformedTrans = rotMatrix.MultiplyVector(headPosition);
//        translation = new Vector3(-transformedTrans.x, transformedTrans.y, transformedTrans.z);

//        // 2. Transform root rotation specially
//        Quaternion rootRot = AxisAngleToQuaternion(poses[0]);
//        Matrix4x4 rootRotMatrix = Matrix4x4.Rotate(rootRot);
//        Matrix4x4 transformedRootMatrix = rotMatrix * rootRotMatrix;
//        Quaternion transformedRootRot = QuaternionFromMatrix(transformedRootMatrix);

//        // Flip Y and Z for handedness
//        transformedRootRot = new Quaternion(
//            transformedRootRot.x,
//            -transformedRootRot.y,
//            -transformedRootRot.z,
//            transformedRootRot.w
//        );

//        // Update root rotation
//        poses[0] = QuaternionToAxisAngle(transformedRootRot);

//        // 3. Negate Y and Z for other joints to match Unity handedness
//        for (int i = 1; i < poses.Length; i++)
//        {
//            // Skip hand joints since they use preset rotations
//            if (i == 20 || i == 21) continue;

//            poses[i] = new Vector3(poses[i].x, -poses[i].y, -poses[i].z);
//        }
//    }

//    #endregion
//}


using UnityEngine;

/// <summary>
/// Optimized coordinate transformations and rotation conversions
/// with reduced allocations and improved performance
/// </summary>
public static class UnityTransformUtils
{
    // Pre-allocated temp arrays for internal use
    private static readonly Vector3[] tempSixD = new Vector3[2];

    #region Rotation Conversions

    /// <summary>
    /// Optimized quaternion to axis-angle conversion
    /// </summary>
    public static Vector3 QuaternionToAxisAngle(Quaternion q)
    {
        // Ensure normalized (in-place)
        float norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        if (norm > 0.0f)
        {
            float invNorm = 1.0f / Mathf.Sqrt(norm);
            q.x *= invNorm;
            q.y *= invNorm;
            q.z *= invNorm;
            q.w *= invNorm;
        }

        float w = Mathf.Clamp(q.w, -1.0f, 1.0f);
        float angle = 2.0f * Mathf.Acos(w);

        // Early exit for near-zero angle
        if (angle < 0.0001f)
            return Vector3.zero;

        float s = Mathf.Sqrt(1.0f - w * w);
        if (s < 0.001f)
        {
            return new Vector3(q.x, q.y, q.z) * angle;
        }

        float invS = angle / s;
        return new Vector3(q.x * invS, q.y * invS, q.z * invS);
    }

    /// <summary>
    /// Optimized axis-angle to quaternion conversion
    /// </summary>
    public static Quaternion AxisAngleToQuaternion(Vector3 axisAngle)
    {
        float angle = axisAngle.magnitude;
        if (angle < 0.0001f)
            return Quaternion.identity;

        float halfAngle = angle * 0.5f;
        float s = Mathf.Sin(halfAngle) / angle;
        return new Quaternion(
            axisAngle.x * s,
            axisAngle.y * s,
            axisAngle.z * s,
            Mathf.Cos(halfAngle)
        );
    }

    /// <summary>
    /// Extract first two columns of rotation matrix as 6D representation
    /// </summary>
    public static Vector3[] MatrixToSixD(Matrix4x4 rotMatrix)
    {
        tempSixD[0] = new Vector3(rotMatrix.m00, rotMatrix.m10, rotMatrix.m20);
        tempSixD[1] = new Vector3(rotMatrix.m01, rotMatrix.m11, rotMatrix.m21);
        return tempSixD;
    }

    /// <summary>
    /// Optimized 6D to matrix conversion using Gram-Schmidt
    /// </summary>
    public static Matrix4x4 SixDToMatrix(Vector3 col1, Vector3 col2)
    {
        // Normalize first column
        float norm1 = col1.magnitude;
        if (norm1 < 0.0001f) norm1 = 1.0f;
        Vector3 b1 = col1 / norm1;

        // Gram-Schmidt for second column
        float dot = b1.x * col2.x + b1.y * col2.y + b1.z * col2.z;
        Vector3 b2 = col2 - dot * b1;

        float norm2 = b2.magnitude;
        if (norm2 < 0.0001f) norm2 = 1.0f;
        b2 /= norm2;

        // Cross product for third column
        Vector3 b3 = new Vector3(
            b1.y * b2.z - b1.z * b2.y,
            b1.z * b2.x - b1.x * b2.z,
            b1.x * b2.y - b1.y * b2.x
        );

        return new Matrix4x4(
            new Vector4(b1.x, b1.y, b1.z, 0),
            new Vector4(b2.x, b2.y, b2.z, 0),
            new Vector4(b3.x, b3.y, b3.z, 0),
            new Vector4(0, 0, 0, 1)
        );
    }

    /// <summary>
    /// Direct 6D to axis-angle conversion
    /// </summary>
    public static Vector3 SixDToAxisAngle(Vector3 col1, Vector3 col2)
    {
        Matrix4x4 matrix = SixDToMatrix(col1, col2);

        // Direct matrix to axis-angle (avoiding quaternion intermediate)
        float trace = matrix.m00 + matrix.m11 + matrix.m22;
        float angle = Mathf.Acos(Mathf.Clamp((trace - 1.0f) * 0.5f, -1.0f, 1.0f));

        if (angle < 0.0001f)
            return Vector3.zero;

        float k = 0.5f / Mathf.Sin(angle);
        return new Vector3(
            (matrix.m21 - matrix.m12) * k * angle,
            (matrix.m02 - matrix.m20) * k * angle,
            (matrix.m10 - matrix.m01) * k * angle
        );
    }

    #endregion

    #region Coordinate Transformations

    /// <summary>
    /// Optimized Unity to AMASS/ML coordinate transformation
    /// </summary>
    public static void TransformCoordinatesToAMASS(
        Vector3[] positions,
        Quaternion[] rotations,
        out Vector3[] transformedPositions,
        out Quaternion[] transformedRotations)
    {
        transformedPositions = new Vector3[positions.Length];
        transformedRotations = new Quaternion[rotations.Length];

        // Pre-compute rotation matrices
        Quaternion coordRot = Quaternion.Euler(90, 0, 0);
        Quaternion leftHandOffset = Quaternion.Euler(180, 0, 0);

        for (int i = 0; i < positions.Length; i++)
        {
            // Position: negate X then rotate
            Vector3 flipped = new Vector3(-positions[i].x, positions[i].y, positions[i].z);
            transformedPositions[i] = coordRot * flipped;

            // Rotation: flip Y,Z then apply transforms
            Quaternion flippedRot = new Quaternion(
                rotations[i].x,
                -rotations[i].y,
                -rotations[i].z,
                rotations[i].w
            );

            if (i == 1) // Left hand
                transformedRotations[i] = coordRot * flippedRot * leftHandOffset;
            else
                transformedRotations[i] = coordRot * flippedRot;
        }
    }

    /// <summary>
    /// Optimized head space transformation
    /// </summary>
    public static void TransformToHeadSpace(
        Vector3 headPos,
        Quaternion headRot,
        Vector3[] handPositions,
        Quaternion[] handRotations,
        out Vector3[] localPositions,
        out Quaternion[] localRotations)
    {
        localPositions = new Vector3[handPositions.Length];
        localRotations = new Quaternion[handRotations.Length];

        Quaternion invHeadRot = Quaternion.Inverse(headRot);

        for (int i = 0; i < handPositions.Length; i++)
        {
            localPositions[i] = invHeadRot * (handPositions[i] - headPos);
            localRotations[i] = invHeadRot * handRotations[i];
        }
    }

    /// <summary>
    /// Direct feature vector creation without intermediate allocations
    /// </summary>
    public static float[] CreateHeadspaceFeatures(
        Vector3[] globalPos,
        Vector3[] globalPosVel,
        Vector3[] globalRot6D,
        Vector3[] globalRotVel6D,
        Vector3[] localHandPos,
        Vector3[] localPosVel,
        Vector3[] localRot6D,
        Vector3[] localRotVel6D)
    {
        float[] features = new float[90];
        int idx = 0;

        // Pack all vectors directly
        for (int i = 0; i < 6; i++) PackVector3(globalRot6D[i], features, ref idx);
        for (int i = 0; i < 6; i++) PackVector3(globalRotVel6D[i], features, ref idx);
        for (int i = 0; i < 3; i++) PackVector3(globalPos[i], features, ref idx);
        for (int i = 0; i < 3; i++) PackVector3(globalPosVel[i], features, ref idx);
        for (int i = 0; i < 4; i++) PackVector3(localRot6D[i], features, ref idx);
        for (int i = 0; i < 4; i++) PackVector3(localRotVel6D[i], features, ref idx);
        for (int i = 0; i < 2; i++) PackVector3(localHandPos[i], features, ref idx);
        for (int i = 0; i < 2; i++) PackVector3(localPosVel[i], features, ref idx);

        return features;
    }

    /// <summary>
    /// Inline vector packing helper
    /// </summary>
    private static void PackVector3(Vector3 v, float[] array, ref int index)
    {
        array[index++] = v.x;
        array[index++] = v.y;
        array[index++] = v.z;
    }

    /// <summary>
    /// Optimized SMPL to Unity conversion
    /// </summary>
    public static void ConvertSMPLToUnity(Vector3[] poses, Vector3 headPosition, ref Vector3 translation)
    {
        // Pre-compute rotation
        Quaternion invCoordRot = Quaternion.Euler(-90, 0, 0);

        // Transform translation
        Vector3 transformed = invCoordRot * headPosition;
        translation = new Vector3(-transformed.x, transformed.y, transformed.z);

        // Transform root specially
        Quaternion rootQuat = AxisAngleToQuaternion(poses[0]);
        rootQuat = invCoordRot * rootQuat;
        rootQuat = new Quaternion(rootQuat.x, -rootQuat.y, -rootQuat.z, rootQuat.w);
        poses[0] = QuaternionToAxisAngle(rootQuat);

        // Flip Y,Z for the joint-local rotations (mirror SMPL-X -> Unity), wrists included;
        // rest orientations such as the OVR-rig left wrist are applied by the motion controller
        for (int i = 1; i < poses.Length; i++)
        {
            poses[i].y = -poses[i].y;
            poses[i].z = -poses[i].z;
        }
    }

    #endregion
}
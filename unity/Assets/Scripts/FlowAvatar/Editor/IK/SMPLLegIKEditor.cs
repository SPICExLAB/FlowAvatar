#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(SMPLLegIK))]
public class SMPLLegIKEditor : Editor
{
    private SMPLLegIK legIK;

    private void OnEnable()
    {
        legIK = (SMPLLegIK)target;
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        EditorGUILayout.Space(5);
        EditorGUILayout.LabelField("SMPLLegIK - Foot Penetration Solver", EditorStyles.boldLabel);
        EditorGUILayout.HelpBox("This component handles leg IK for foot penetration. All parameters are managed through the SMPLXFootPlacer component.", MessageType.Info);

        EditorGUILayout.Space(10);

        // Joint references section
        EditorGUILayout.LabelField("Joint References", EditorStyles.boldLabel);
        GUI.enabled = false; // Make these fields read-only
        EditorGUILayout.ObjectField("Hip Joint", legIK.hipJoint, typeof(Transform), true);
        EditorGUILayout.ObjectField("Knee Joint", legIK.kneeJoint, typeof(Transform), true);
        EditorGUILayout.ObjectField("Ankle Joint", legIK.ankleJoint, typeof(Transform), true);
        EditorGUILayout.ObjectField("Foot Joint", legIK.footJoint, typeof(Transform), true);
        GUI.enabled = true;

        // Debug settings
        EditorGUILayout.Space(10);
        EditorGUILayout.PropertyField(serializedObject.FindProperty("showDebugInfo"));

        // Runtime stats
        if (Application.isPlaying && legIK.IsJointChainValid())
        {
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Runtime Statistics", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Max Penetration Amount", $"{legIK.GetLastCorrectionAmount():F3}");
            EditorGUILayout.LabelField("Target Reached", legIK.WasTargetReached() ? "Yes" : "No");
            EditorGUILayout.LabelField("Forward Step Pose", legIK.IsForwardStepPose() ? "Yes" : "No");

            // Current joint rotations
            EditorGUILayout.Space(5);
            EditorGUILayout.LabelField("Current Joint Rotations (X-Axis)", EditorStyles.boldLabel);
            if (legIK.hipJoint != null)
                EditorGUILayout.LabelField("Hip", $"{NormalizeAngle(legIK.hipJoint.localEulerAngles.x):F1}°");
            if (legIK.kneeJoint != null)
                EditorGUILayout.LabelField("Knee", $"{NormalizeAngle(legIK.kneeJoint.localEulerAngles.x):F1}°");
            if (legIK.ankleJoint != null)
                EditorGUILayout.LabelField("Ankle", $"{NormalizeAngle(legIK.ankleJoint.localEulerAngles.x):F1}°");
            if (legIK.footJoint != null)
                EditorGUILayout.LabelField("Foot", $"{NormalizeAngle(legIK.footJoint.localEulerAngles.x):F1}°");
        }

        serializedObject.ApplyModifiedProperties();
    }

    private void OnSceneGUI()
    {
        if (!Application.isPlaying || legIK == null || !legIK.showDebugInfo)
            return;

        if (legIK.IsJointChainValid())
        {
            // Draw the bone chain
            Handles.color = Color.white;
            Handles.DrawLine(legIK.hipJoint.position, legIK.kneeJoint.position);
            Handles.DrawLine(legIK.kneeJoint.position, legIK.ankleJoint.position);
            Handles.DrawLine(legIK.ankleJoint.position, legIK.footJoint.position);

            // Draw rotation axes
            Handles.color = Color.red;
            Handles.DrawLine(legIK.hipJoint.position, legIK.hipJoint.position + legIK.hipJoint.right * 0.1f);
            Handles.color = Color.green;
            Handles.DrawLine(legIK.kneeJoint.position, legIK.kneeJoint.position + legIK.kneeJoint.right * 0.1f);
            Handles.color = Color.blue;
            Handles.DrawLine(legIK.ankleJoint.position, legIK.ankleJoint.position + legIK.ankleJoint.right * 0.1f);

            // Draw the target position
            Vector3 targetPos = legIK.GetTargetPosition();
            if (targetPos != Vector3.zero)
            {
                Handles.color = legIK.WasTargetReached() ? Color.green : Color.red;
                Handles.SphereHandleCap(0, targetPos, Quaternion.identity, 0.03f, EventType.Repaint);
                Handles.DrawLine(legIK.footJoint.position, targetPos);
            }
        }
    }

    private float NormalizeAngle(float angle)
    {
        while (angle > 180f)
            angle -= 360f;
        while (angle < -180f)
            angle += 360f;
        return angle;
    }
}
#endif
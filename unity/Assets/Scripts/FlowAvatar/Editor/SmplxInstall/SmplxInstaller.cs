using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Copies the SMPL-X files FlowAvatar needs from the official SMPL-X Unity add-on
/// (downloaded by the user from https://smpl-x.is.tue.mpg.de) into Assets/SMPLX-OVR.
/// Each file is written together with the .meta template shipped in MetaTemplates, so
/// it gets the GUID that the avatar prefab and the demo scene reference. (Unity deletes
/// a .meta whose asset is missing, so the repository cannot keep these .meta files
/// next to the absent SMPL-X files.)
/// </summary>
public static class SmplxInstaller
{
    const string TargetFolder = "Assets/SMPLX-OVR/";
    const string TemplateFolder = "Assets/Scripts/FlowAvatar/Editor/SmplxInstall/MetaTemplates/";

    // (path inside the add-on's Assets/SMPLX folder, path inside TargetFolder, required)
    static readonly (string source, string target, bool required)[] Files =
    {
        ("Models/smplx-neutral.fbx", "smplx-neutral.fbx", true),
        ("Resources/smplx_betas_to_joints_female.json", "Resources/smplx_betas_to_joints_female.json", true),
        ("Resources/smplx_betas_to_joints_male.json", "Resources/smplx_betas_to_joints_male.json", true),
        ("Resources/smplx_betas_to_joints_neutral.json", "Resources/smplx_betas_to_joints_neutral.json", true),
        ("Textures/smplx_texture_f_alb.png", "Textures/smplx_texture_f_alb.png", false),
        ("Textures/smplx_texture_m_alb.png", "Textures/smplx_texture_m_alb.png", false),
    };

    [MenuItem("FlowAvatar/Install SMPL-X Files...")]
    public static void InstallFromDialog()
    {
        string folder = EditorUtility.OpenFolderPanel(
            "Select the SMPL-X Unity add-on (its project folder or Assets/SMPLX)", "", "");
        if (string.IsNullOrEmpty(folder))
            return;
        string error = Install(folder);
        EditorUtility.DisplayDialog("FlowAvatar", error ?? "SMPL-X files installed. The avatar is ready.", "OK");
    }

    /// <summary>Batch mode: -executeMethod SmplxInstaller.InstallBatch -smplxSource &lt;folder&gt;</summary>
    public static void InstallBatch()
    {
        string[] args = Environment.GetCommandLineArgs();
        int i = Array.IndexOf(args, "-smplxSource");
        string error = i >= 0 && i + 1 < args.Length ? Install(args[i + 1]) : "missing -smplxSource <folder>";
        if (error != null)
        {
            Debug.LogError("[SmplxInstaller] " + error);
            EditorApplication.Exit(1);
        }
    }

    /// <returns>null on success, otherwise an error message</returns>
    public static string Install(string folder)
    {
        string addon = FindAddonFolder(folder);
        if (addon == null)
            return $"No SMPL-X add-on found in {folder}. Select the folder that contains Models/smplx-neutral.fbx " +
                   "(e.g. SMPLX-Unity/Assets/SMPLX).";

        foreach (var f in Files.Where(f => f.required))
            if (!File.Exists(Path.Combine(addon, f.source)))
                return $"{f.source} is missing in {addon}.";

        foreach (var f in Files)
        {
            string source = Path.Combine(addon, f.source);
            if (!File.Exists(source))
                continue;
            string target = TargetFolder + f.target;
            // Replace any earlier copy so it takes the expected GUID
            if (File.Exists(target))
                AssetDatabase.DeleteAsset(target);
            Directory.CreateDirectory(Path.GetDirectoryName(target));
            File.Copy(source, target, true);
            File.Copy(TemplateFolder + Path.GetFileName(f.target) + ".meta.txt", target + ".meta", true);
        }
        AssetDatabase.Refresh();

        foreach (var f in Files)
        {
            string target = TargetFolder + f.target;
            if (!File.Exists(target))
                continue;
            string expected = Regex.Match(File.ReadAllText(TemplateFolder + Path.GetFileName(f.target) + ".meta.txt"),
                                          @"guid: (\w+)").Groups[1].Value;
            if (AssetDatabase.AssetPathToGUID(target) != expected)
                return $"{target} was imported with an unexpected GUID; delete it and run the installer again.";
        }
        Debug.Log($"[SmplxInstaller] Installed the SMPL-X files from {addon} into {TargetFolder}");
        return null;
    }

    static string FindAddonFolder(string folder)
    {
        foreach (string candidate in new[] { folder, Path.Combine(folder, "SMPLX"), Path.Combine(folder, "Assets", "SMPLX"),
                                             Path.Combine(folder, "SMPLX-Unity", "Assets", "SMPLX") })
            if (File.Exists(Path.Combine(candidate, "Models", "smplx-neutral.fbx")))
                return candidate;
        return null;
    }
}

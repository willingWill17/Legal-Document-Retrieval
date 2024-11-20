from sentence_transformers import CrossEncoder
import numpy as np

# Ensure numbers are printed in standard decimal format
np.set_printoptions(suppress=True)

query = "Sản phẩm phần mềm có được hưởng ưu đãi về thời gian miễn thuế, giảm thuế hay không? Nếu được thì trong vòng bao nhiêu năm?"
# query = "Nhân sự của Ban quản lý dự án có phải do Giám đốc Ban quản lý dự án chương trình, dự án sử dụng vốn ODA, vốn vay ưu đãi tuyển chọn không?"

senteces = [
    "Điều 20. Ưu đãi về thời gian miễn thuế, giảm thuế\n1. Miễn thuế bốn năm, giảm 50% số thuế phải nộp trong chín năm tiếp theo đối với:\na) Thu nhập của doanh nghiệp từ thực hiện dự án đầu tư quy định tại khoản 1 Điều 19 Thông tư số 78/2014/TT-BTC (được sửa đổi, bổ sung tại Khoản 1 Điều 11 Thông tư này).",
    "Điều kiện được hưởng\nCán bộ quản lý, giáo viên, nhân viên được hưởng chính sách khi bảo đảm các điều kiện sau:\n1. Là người đang làm việc tại cơ sở giáo dục ngoài công lập trước khi cơ sở phải tạm dừng hoạt động theo yêu cầu của cơ quan nhà nước có thẩm quyền để phòng, chống dịch COVID-19 tính từ ngày 01 tháng 5 năm 2021 đến hết ngày 31 tháng 12 năm 2021.\n2. Nghỉ việc không hưởng lương từ 01 tháng trở lên tính từ ngày 01 tháng 5 năm 2021 đến hết ngày 31 tháng 12 năm 2021.\n3. Chưa được hưởng chính sách hỗ trợ đối với người lao động tạm hoãn hợp đồng lao động, nghỉ việc không hưởng lương theo quy định tại khoản 4, khoản 5, khoản 6 Mục II Nghị quyết số 68/NQ-CP ngày 01 tháng 7 năm 2021 của Chính phủ về một số chính sách hỗ trợ người lao động và người sử dụng lao động gặp khó khăn do đại dịch COVID-19, Nghị quyết số 126/NQ-CP ngày 08 tháng 10 năm 2021 của Chính phủ sửa đổi, bổ sung Nghị quyết số 68/NQ-CP ngày 01 tháng 7 năm 2021 của Chính phủ về một số chính sách hỗ trợ người lao động và người sử dụng lao động gặp khó khăn do đại dịch COVID-19 (sau đây gọi tắt là Nghị quyết số 68/NQ-CP) do không tham gia Bảo hiểm xã hội bắt buộc.\n4. Có xác nhận làm việc tại cơ sở giáo dục ngoài công lập ít nhất hết năm học 2021 - 2022 theo kế hoạch năm học của địa phương, bao gồm cơ sở giáo dục ngoài công lập đã làm việc trước đây hoặc cơ sở giáo dục ngoài công lập khác trong trường hợp cơ sở giáo dục ngoài công lập trước đây làm việc không hoạt động trở lại.",
    "Điều 3. Giải thích từ ngữ\nKinh doanh vận tải bằng xe ô tô là việc thực hiện ít nhất một trong các công đoạn chính của hoạt động vận tải (trực tiếp điều hành phương tiện, lái xe hoặc quyết định giá cước vận tải) để vận chuyển hành khách, hàng hóa trên đường bộ nhằm mục đích sinh lợi.",
    "Lãnh đạo Cục\n1. Cục Thuế doanh nghiệp lớn có Cục trưởng và không quá 03 Phó Cục trưởng; Cục trưởng chịu trách nhiệm trước Tổng cục trưởng Tổng cục Thuế và trước pháp luật về toàn bộ hoạt động của Cục.\nPhó Cục trưởng chịu trách nhiệm trước Cục trưởng và trước pháp luật về nhiệm vụ được phân công.\n2. Việc bổ nhiệm, miễn nhiệm, cách chức Cục trưởng, Phó Cục trưởng và các chức danh lãnh đạo khác của Cục Thuế doanh nghiệp lớn thực hiện theo quy định của pháp luật và phân cấp quản lý cán bộ của Bộ Tài chính.",
    "Quy chế tổ chức và hoạt động của Ban quản lý dự án\n...\n4. Nhân sự của Ban quản lý dự án do Giám đốc Ban quản lý dự án tuyển chọn, bổ nhiệm và miễn nhiệm. Chức năng, nhiệm vụ, quyền hạn, chế độ (lương, thưởng, phụ cấp,...) được quy định cụ thể trong điều khoản giao việc phù hợp với vị trí công tác và quy định của pháp luật có liên quan. Việc lựa chọn, thuê tuyển, điều động cán bộ không thuộc biên chế của chủ dự án hoặc cơ quan chủ quản làm việc cho Ban quản lý dự án thực hiện theo nội dung văn kiện chương trình, dự án đã được cấp có thẩm quyền phê duyệt và phù hợp với quy định pháp luật có liên quan."
]
tokenized_pairs = [[query, sent] for sent in senteces]
# model_id = 'itdainb/PhoRanker'
model_id = '/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/phoranker_result/model_epoch_1_step_186.bin'
model = CrossEncoder(model_id, device='cuda')

# print(model.rank(query,senteces,top_k=4,return_documents=True))
scores = model.predict(tokenized_pairs)

print(scores)
